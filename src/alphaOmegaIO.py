#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 20:19:44 2020

@author: greydon
"""
import numpy as np
import os
import struct
import string
import datetime
import regex as re

max_string_len = '32s'

# fid.tell()
# fid.seek(228)
# buf=fid.read(38)
# key=Type0_SetBoards[3][0]
# fmt=Type0_SetBoards[3][1]

# fmtt = '<' + '18s'
# buf = fid.read(struct.calcsize(fmtt))
# val=struct.unpack(fmtt, buf)
class HeaderReader:
	def __init__(self, fid, description):
		self.fid = fid
		self.description = description
		self.ignore_bytes={'å?','\\L',':'}
		self.split_bytes={'\x00N','\x00AL','(',','}
	
	def read_f(self, offset=None):
		if offset is not None:
			self.fid.seek(offset)
		
		nextBlockCount=[item[1] for item in self.description if item[0] == 'checkBlockCount']
		self.description=[item for item in self.description if item[0] != 'checkBlockCount']

		d = {}
		for key, fmt in self.description:
			skipKey=False
			if '{}' in fmt:
				fmtt = '<'+fmt.format(d['m_nextBlock']-self.fid.tell())
			elif any(x in key for x in {'m_mapVersion','m_ApplicationName','m_ResourceVersion','m_Reserved'}):
				if d['m_version'] >=2:
					fmtt = '<' + fmt  # insures use of standard sizes
				else:
					skipKey=True
			else:
				fmtt = '<' + fmt  # insures use of standard sizes
			
			if not skipKey:
				buf = self.fid.read(struct.calcsize(fmtt))
				if len(buf) != struct.calcsize(fmtt):
					return None
				
				val = list(struct.unpack(fmtt, buf))
				for i, ival in enumerate(val):
					if hasattr(ival, 'split'):
						ival=ival.decode('ISO-8859-1').replace('/[^\x00-\x7F]/g', '')
						for ibyte in self.ignore_bytes:
							ival=ival.replace(ibyte,'')
						for ibyte in re.findall("(\x00[\d]+)", ival):
							ival=ival.replace(ibyte,'')
						for ibyte in re.findall("(\x00[a-zA-z]+[\d]+)", ival):
							ival=ival.replace(ibyte,'')
						for ibyte in re.findall("(\x00[a-zA-z]+)", ival):
							ival=ival.replace(ibyte,'')
						for ibyte in re.findall("(\x00 [a-zA-z]+)", ival):
							ival=ival.replace(ibyte,'')
						
						for ibyte in self.split_bytes:
							ival=ival.split(ibyte)[0]
							
						val[i] = re.sub(r'\p{P}', lambda m: "/" if m.group(0) == "/" else " ",''.join(filter(lambda x: x in string.printable, ival)).split('|')[-1]).strip()
				
				if len(val) == 1:
					val = val[0]
				d[key] = val
		
		if not d:
			return d
		elif 'm_isAnalog' in list(d) and len(d)==6:
			return d
		else:
			if nextBlockCount:
				if not nextBlockCount==self.fid.tell():
					self.fid.seek(nextBlockCount[0])
			else:
				if not d['m_nextBlock']==self.fid.tell():
					self.fid.seek(d['m_nextBlock'])
			
			return d

class alphaOmegaIO:
	"""
	Class for reading data from Alpha Omega .mpx files
	"""
	def __init__(self, filename=None, channelsKeep=None):
		"""
		Arguments:
			filename : the .map Alpha Omega file name
			"""
		super(alphaOmegaIO, self).__init__()
		self.filename = filename
	
	def block_data(self, chansKeep=None):
		
		sideDict={'rt':'right',
				  'lt':'left'}
		def count_samples(m_length):
			"""
			Count the number of signal samples available in a type 5 data block
			of length m_length
			"""
		
			# for information about type 5 data block, see [1]
			count = int((m_length-6)/2-2)
			# -6 corresponds to the header of block 5, and the -2 take into
			# account the fact that last 2 values are not available as the 4
			# corresponding bytes are coding the time stamp of the beginning
			# of the block
			return count

		fid = open(self.filename, 'rb')
# 		fid = open(filename, 'rb')
# 		fid.close()
# 		fid.seek(192)
		
		pos_block = 0  # position of the current block in the file
		file_blocks = []  # list of data blocks available in the file
		while True:
			first_4_bytes = fid.read(4)
			if len(first_4_bytes) < 4:
				# we have reached the end of the file
				break
			else:
				m_length, m_TypeBlock = struct.unpack('Hcx', first_4_bytes)
			
			description=dict_header_type.get(bytes([ (p,0)[p == 0xFF] for p in m_TypeBlock ]).decode("utf-8"), Type_Unknown).copy()
			
			if 'm_nextBlock' not in description:
				description.append(('checkBlockCount',pos_block+m_length))
			
			block = HeaderReader(fid, description).read_f()
			block.update({'m_length': m_length,
						  'm_TypeBlock': bytes([ (p,0)[p == 0xFF] for p in m_TypeBlock ]).decode("utf-8"),
						  'pos': pos_block})
			
			if block['m_TypeBlock'] == '2':
				type_subblock = 'unknown_channel_type(m_Mode=' \
										+ str(block['m_Mode']) + ')'
				if block['m_isAnalog'] == 0:
					# digital channel
					type_subblock = 'digital'
					description = Type2_SubBlockDigitalChannels.copy()
					for i,e in enumerate(description):
							if e[0]=='m_Name': 
								temp=list(description[i])
								temp[1]=str(m_length-30)+'s'
								description[i]=tuple(temp)
					description.append(('checkBlockCount',block['m_nextBlock']))
				elif block['m_isAnalog'] == 1:
					# analog channel
					if block['m_Mode'] == 1:
						# level channel
						type_subblock = 'level'
						description = Type2_SubBlockLevelChannels.copy()
						for i,e in enumerate(description):
							if e[0]=='m_Name': 
								temp=list(description[i])
								temp[1]=str(m_length-48)+'s'
								description[i]=tuple(temp)
					elif block['m_Mode'] == 2:
						# external trigger channel
						type_subblock = 'external_trigger'
						description = Type2_SubBlockExtTriggerChannels.copy()
						for i,e in enumerate(description):
							if e[0]=='m_Name': 
								temp=list(description[i])
								temp[1]=str(m_length-48)+'s'
								description[i]=tuple(temp)
					else:
						# continuous channel
						type_subblock = 'continuous(Mode' \
										+ str(block['m_Mode']) + ')'
						description = Type2_SubBlockContinuousChannels.copy()
						for i,e in enumerate(description):
							if e[0]=='m_Name': 
								temp=list(description[i])
								temp[1]=str(m_length-38)+'s'
								description[i]=tuple(temp)
						
					description.append(('checkBlockCount',block['m_nextBlock']))
				
				subblock = HeaderReader(fid, description).read_f()
				
				if 'm_Name' not in list(subblock):
					subblock.update({'m_Name': 'unknown_name'})
				block.update(subblock)
				block.update({'type_subblock': type_subblock})
			
			file_blocks.append(block)
			pos_block += m_length
			fid.seek(pos_block)
		
		# step 2: find the available channels
		list_chan = []  # list containing indexes of channel blocks
		chan_labels=[]
		for ind_block, block in enumerate(file_blocks):
			if block['m_TypeBlock'] == '2':
				list_chan.append(ind_block)
				chan_labels.append(block['m_Name'])
		
		channelsKeep={}
		if chansKeep is None:
			for ichan in np.unique(chan_labels):
				channelsKeep[ichan]=ichan
		else:
			for ichan in chansKeep:
				channelsKeep[ichan.split('/')[0].strip()]=ichan
			
		# step 3: find blocks containing data for the available channels
		list_data = []  # list of lists of indexes of data blocks
		# corresponding to each channel
		for ind_chan, chan in enumerate(list_chan):
			list_data.append([])
			num_chan = file_blocks[chan]['m_numChannel']
			for ind_block, block in enumerate(file_blocks):
				if block['m_TypeBlock'] == '5':
					if block['m_numChannel'] == num_chan:
						list_data[ind_chan].append(ind_block)
		
		# step 4: compute the length (number of samples) of the channels
		chan_len = np.zeros(len(list_data), dtype=np.int)
		for ind_chan, list_blocks in enumerate(list_data):
			for ind_block in list_blocks:
				chan_len[ind_chan] += count_samples(
					file_blocks[ind_block]['m_length'])
		
		# step 5: find channels for which data are available
		ind_valid_chan = np.nonzero(chan_len)[0]
		
		# step 6: load the data
		# TODO give the possibility to load data as AnalogSignalArrays
		recordStart=datetime.datetime(file_blocks[0]['m_date_year'], file_blocks[0]['m_date_month'], file_blocks[0]['m_date_day'],
									  file_blocks[0]['m_time_hour'], file_blocks[0]['m_time_minute'], file_blocks[0]['m_time_second'], file_blocks[0]['m_time_hsecond'])
		ana_sig={'file':[],'name':[],'ch_id':[],'sampling_rate':[],'datetime':[],'t_start':[],'data':[]}
		for ind_chan in ind_valid_chan:
			if file_blocks[list_chan[ind_chan]]['m_Name'] in list(channelsKeep.values()) or file_blocks[list_chan[ind_chan]]['m_Name'] in list(channelsKeep):
				list_blocks = list_data[ind_chan]
				ind = 0  # index in the data vector
			
				# read time stamp for the beginning of the signal
				form = '<l'  # reading format
				ind_block = list_blocks[0]
				count = count_samples(file_blocks[ind_block]['m_length'])
				fid.seek(file_blocks[ind_block]['pos'] + 6 + count * 2)
				buf = fid.read(struct.calcsize(form))
				val = struct.unpack(form, buf)
				start_index = val[0]
			
				# WARNING: in the following blocks are read supposing taht they
				# are all contiguous and sorted in time. I don't know if it's
				# always the case. Maybe we should use the time stamp of each
				# data block to choose where to put the read data in the array.
			
				temp_array = np.empty(chan_len[ind_chan], dtype=np.int16)
				# NOTE: we could directly create an empty AnalogSignal and
				# load the data in it, but it is much faster to load data
				# in a temporary numpy array and create the AnalogSignals
				# from this temporary array
				for ind_block in list_blocks:
					count = count_samples(
						file_blocks[ind_block]['m_length'])
					fid.seek(file_blocks[ind_block]['pos'] + 6)
					temp_array[ind:ind + count] = \
						np.fromfile(fid, dtype=np.int16, count=count)
					ind += count
			
				sampling_rate = file_blocks[list_chan[ind_chan]]['m_SampleRate'] * 1000
				t_start = (start_index / sampling_rate)
				
				ana_sig['file'].append(os.path.basename(self.filename))
				ana_sig['name'].append(file_blocks[list_chan[ind_chan]]['m_Name'])
				ana_sig['ch_id'].append(file_blocks[list_chan[ind_chan]]['m_numChannel'])
				ana_sig['sampling_rate'].append(sampling_rate)
				ana_sig['datetime'].append("%s.%03d" % ((recordStart + datetime.timedelta(seconds=t_start)).strftime('%Y-%m-%dT%H:%M:%S'), int(file_blocks[0]['m_time_hsecond']) / 1000))
				ana_sig['t_start'].append(t_start)
				ana_sig['data'].append(temp_array)
		
		fid.close()
		
		header=file_blocks[0]
		header['filename']=self.filename
		header['datetime']="%s.%03d" % ((recordStart + datetime.timedelta(seconds=min(ana_sig['t_start']))).strftime('%Y-%m-%dT%H:%M:%S'), int(file_blocks[0]['m_time_hsecond']) / 1000)
		header['side']=re.findall(r"[a-zA-z]+\d*", os.path.basename(self.filename))[0].lower()
		header['depth']=float(re.findall(r"[-+]?\d*\.\d+|\d+", os.path.basename(self.filename))[1])*-1
		header['sampling_rate']=sampling_rate
		
		return ana_sig, header

#%%

TypeH_Header = [
	('m_nextBlock', 'l'),
	('m_version', 'h'),
	('m_time_hour', 'B'),
	('m_time_minute', 'B'),
	('m_time_second', 'B'),
	('m_time_hsecond', 'B'),
	('m_date_day', 'B'),
	('m_date_month', 'B'),
	('m_date_year', 'H'),
	('m_date_dayofweek', 'B'),
	('blank', 'x'),  # one byte blank because of the 2 bytes alignement
	('m_MinimumTime', 'd'),
	('m_MaximumTime', 'd'),
	('m_EraseCount', 'l'),
	('m_mapVersion', 'b'),
	('m_ApplicationName', '10s'),
	('m_ResourceVersion', '4s'),
	('blank_2', 'x'),
	('m_Reserved', 'l')]

Type0_SetBoards = [
	('m_nextBlock', 'l'),
	('m_BoardCount', 'h'),
	('m_GroupCount', 'h'),
	('m_placeMainWindow', '{}s')]  # WARNING: unknown type ('x' is wrong)

Type1_Boards = [  # WARNING: needs to be checked
	('m_nextBlock', 'l'),		#4
	('m_Number', 'h'),			#6
	('m_countChannel', 'h'),	#8
	('m_countAnIn', 'h'),		#10
	('m_countAnOut', 'h'),		#12
	('m_countDigIn', 'h'),		#14
	('m_countDigOut', 'h'),		#16
	('m_TrigCount', 'h'),		#18	# not defined in 5.3.3 but appears in 5.5.1 and seems to really exist in files, WARNING: check why 'm_TrigCount is not in the C code [2]
	('m_Amplitude', 'f'),		#22
	('m_cSampleRate', 'f'),		#26	# sample rate seems to be given in kHz
	('m_Duration', 'f'),		#30
	('m_nPreTrigmSec', 'f'),	#34
	('m_nPostTrigmSec', 'f'),	#38
	('m_TrgMode', 'h'),			#40
	('m_LevelValue', 'h'),		#42 # after this line, 5.3.3 is wrong,check example in 5.5.1 for the right fields, WARNING: check why the following part is not corrected in the C code [2]
	('m_nSamples', 'h'),		#44
	('m_fRMS', 'f'),			#48
	('m_ScaleFactor', 'f'),		#52
	('m_DapTime', 'f'),			#56
	('m_nameBoard', '{}s')]
# ('m_DiscMaxValue','h'), # WARNING: should this exist?
# ('m_DiscMinValue','h') # WARNING: should this exist?

Type2_DefBlocksChannels = [
    # common parameters for all types of channels
    ('m_nextBlock', 'l'),
    ('m_isAnalog', 'h'),
    ('m_isInput', 'h'),
    ('m_numChannel', 'h'),
    ('m_numColor', 'h'),
    ('m_Mode', 'h')]

Type2_SubBlockContinuousChannels = [
    # continuous channels parameters
    ('blank', '2x'),  # WARNING: this is not in the specs but it seems needed
    ('m_Amplitude', 'f'),
    ('m_SampleRate', 'f'),
    ('m_ContBlkSize', 'h'),
    ('m_ModeSpike', 'h'),  # WARNING: the C code [2] uses usigned short here
    ('m_Duration', 'f'),
    ('m_bAutoScale', 'h'),
    ('m_Name', '{}s')]

Type2_SubBlockLevelChannels = [  # WARNING: untested
    # level channels parameters
    ('m_Amplitude', 'f'),
    ('m_SampleRate', 'f'),
    ('m_nSpikeCount', 'h'),
    ('m_ModeSpike', 'h'),
    ('m_nPreTrigmSec', 'f'),
    ('m_nPostTrigmSec', 'f'),
    ('m_LevelValue', 'h'),
    ('m_TrgMode', 'h'),
    ('m_YesRms', 'h'),
    ('m_bAutoScale', 'h'),
    ('m_Name', '{}s')]

Type2_SubBlockExtTriggerChannels = [  # WARNING: untested
    # external trigger channels parameters
    ('m_Amplitude', 'f'),
    ('m_SampleRate', 'f'),
    ('m_nSpikeCount', 'h'),
    ('m_ModeSpike', 'h'),
    ('m_nPreTrigmSec', 'f'),
    ('m_nPostTrigmSec', 'f'),
    ('m_TriggerNumber', 'h'),
    ('m_Name', '{}s')]

Type2_SubBlockDigitalChannels = [
    # digital channels parameters
    ('m_SampleRate', 'f'),
    ('m_SaveTrigger', 'h'),
    ('m_Duration', 'f'),
    ('m_PreviousStatus', 'h'),  # WARNING: check difference with C code here
    ('m_Name', '{}s')]

Type2_SubBlockUnknownChannels = [
    # WARNING: We have a mode that doesn't appear in our spec, so we don't
    # know what are the fields.
    # It seems that for non-digital channels the beginning is
    # similar to continuous channels. Let's hope we're right...
    ('blank', '2x'),
    ('m_Amplitude', 'f'),
    ('m_SampleRate', 'f')]
# there are probably other fields after...

Type6_DefBlockTrigger = [  # WARNING: untested
    ('m_nextBlock', 'l'),
    ('m_Number', 'h'),
    ('m_countChannel', 'h'),
    ('m_StateChannels', 'i'),
    ('m_numChannel1', 'h'),
    ('m_numChannel2', 'h'),
    ('m_numChannel3', 'h'),
    ('m_numChannel4', 'h'),
    ('m_numChannel5', 'h'),
    ('m_numChannel6', 'h'),
    ('m_numChannel7', 'h'),
    ('m_numChannel8', 'h'),
    ('m_Name', 'c')]

Type3_DefBlockGroup = [  # WARNING: untested
    ('m_nextBlock', 'l'),
    ('m_Number', 'h'),
    ('m_Z_Order', 'h'),
    ('m_countSubGroups', 'h'),
    ('m_placeGroupWindow', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_NetLoc', 'h'),
    ('m_locatMax', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_nameGroup', 'c')]

Type4_DefBlockSubgroup = [  # WARNING: untested
    ('m_nextBlock', 'l'),
    ('m_Number', 'h'),
    ('m_TypeOverlap', 'h'),
    ('m_Z_Order', 'h'),
    ('m_countChannel', 'h'),
    ('m_NetLoc', 'h'),
    ('m_location', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_bIsMaximized', 'h'),
    ('m_numChannel1', 'h'),
    ('m_numChannel2', 'h'),
    ('m_numChannel3', 'h'),
    ('m_numChannel4', 'h'),
    ('m_numChannel5', 'h'),
    ('m_numChannel6', 'h'),
    ('m_numChannel7', 'h'),
    ('m_numChannel8', 'h'),
    ('m_Name', 'c')]

Type5_DataBlockOneChannel = [
    ('m_numChannel', 'h')]
# WARNING: 'm_numChannel' (called 'm_Number' in 5.4.1 of [1]) is supposed
# to be uint according to 5.4.1 but it seems to be a short in the files
# (or should it be ushort ?)

# WARNING: In 5.1.1 page 121 of [1], they say "Note: 5 is used for demo
# purposes, 7 is used for real data", but looking at some real datafiles,
# it seems that block of type 5 are also used for real data...

Type7_DataBlockMultipleChannels = [  # WARNING: unfinished
    ('m_lenHead', 'h'),  # WARNING: unknown true type
    ('FINT', 'h')]
# WARNING: there should be data after...

TypeP_DefBlockPeriStimHist = [  # WARNING: untested
    ('m_Number_Chan', 'h'),
    ('m_Position', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible', 'h'),
    ('m_DurationSec', 'f'),
    ('m_Rows', 'i'),
    ('m_DurationSecPre', 'f'),
    ('m_Bins', 'i'),
    ('m_NoTrigger', 'h')]

TypeF_DefBlockFRTachogram = [  # WARNING: untested
    ('m_Number_Chan', 'h'),
    ('m_Position', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible', 'h'),
    ('m_DurationSec', 'f'),
    ('m_AutoManualScale', 'i'),
    ('m_Max', 'i')]

TypeR_DefBlockRaster = [  # WARNING: untested
    ('m_Number_Chan', 'h'),
    ('m_Position', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible', 'h'),
    ('m_DurationSec', 'f'),
    ('m_Rows', 'i'),
    ('m_NoTrigger', 'h')]

TypeI_DefBlockISIHist = [  # WARNING: untested
    ('m_Number_Chan', 'h'),
    ('m_Position', 'x'),  # WARNING: unknown type ('x' is wrong)
    ('m_isStatVisible', 'h'),
    ('m_DurationSec', 'f'),
    ('m_Bins', 'i'),
    ('m_TypeScale', 'i')]

Type8_MarkerBlock = [  # WARNING: untested
    ('m_Number_Channel', 'h'),
    ('m_Time', 'l')]  # WARNING: check what's the right type here.
# It seems that the size of time_t type depends on the system typedef,
# I put long here but I couldn't check if it is the right type

Type9_ScaleBlock = [  # WARNING: untested
    ('m_Number_Channel', 'h'),
    ('m_Scale', 'f')]

Type_Unknown = []

dict_header_type = {
    'h': TypeH_Header,
    '0': Type0_SetBoards,
    '1': Type1_Boards,
    '2': Type2_DefBlocksChannels,
    '6': Type6_DefBlockTrigger,
    '3': Type3_DefBlockGroup,
    '4': Type4_DefBlockSubgroup,
    '5': Type5_DataBlockOneChannel,
    '7': Type7_DataBlockMultipleChannels,
    'P': TypeP_DefBlockPeriStimHist,
    'F': TypeF_DefBlockFRTachogram,
    'R': TypeR_DefBlockRaster,
    'I': TypeI_DefBlockISIHist,
    '8': Type8_MarkerBlock,
    '9': Type9_ScaleBlock
}


#%%

# filename=r'/media/veracrypt6/projects/mer_analysis/mer/source/sub-P156/LT1D0.013F0001.mpx'
# file_in=alphaOmegaIO(r'/media/veracrypt6/projects/mer_analysis/mer/source/sub-P154/LT1D0.016F0001.mpx')
# file_in=alphaOmegaIO(r'/media/veracrypt6/projects/mer_analysis/mer/source/raw/sub-P189/RT1D2.992F0001.mpx')


# file_data,header=file_in.block_data(chansKeep={'RAW 01 / Central','RAW 02 / Anterior','RAW 03 / Posterior',
# 												  'RAW 04 / Medial','RAW 05 / Lateral'})




