# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 01:43:53 2020

@author: greydon
"""
import os
import numpy as np
import pandas as pd
pd.set_option('precision', 6)
from collections import OrderedDict
import datetime
import shutil
import regex as re
import struct
from ext_libs.edflibpy import EDFwriter
import scipy.io as spio
from PySide2 import QtGui, QtCore, QtWidgets

from alphaOmegaIO import alphaOmegaIO
from helpers import bidsHelper, sorted_nicely, EDFReader, sec2time


mer_file_metadata = {
	'Experimenter': ['Greydon Gilmore'],
	'Lab': 'Jog Lab',
	'InstitutionName': 'Western University',
	'InstitutionAddress':'339 Windermere Rd',
	'ExperimentDescription': '',
	'DatasetName': 'MERIntraop'
}

leadpoint_info = {
		'Manufacturer': 'Medtronic',
		'ManufacturersModelName': 'Leadpoint 5',
		'SamplingFrequency': 24000,
		'LeftSideLabels': {'_25_','_19_', '_01_01_all'},
		'RightSideLabels': {'_26_','_20_', '_02_01_all'},
		'HighpassFilter': 500,
		'LowpassFilter': 5000,
		'MERUnit': 'μV',
		'MERchannels':{
			'cen': 1,
			'ant': 2,
			'pos': 3,
			'med': 4,
			'lat': 5
		},
		'MERchannelsMap':{
			'center':'cen',
			 'anterior':'ant',
			 'posterior':'pos',
			 'medial':'med',
			 'lateral':'lat'
		},
		'PowerLineFrequency': 60,
		'RecordingType': 'epoched',
		'ElectrodeInfo': {
			'Manufacturer': 'FHC',
			'Type': 'semi-micro',
			'Material': 'Tungsten',
			'Diameter': 100
		}
}

alphaomega_info = {
		'Manufacturer': 'Alpha Omega',
		'ManufacturersModelName': 'Neuro Omega',
		'SamplingFrequency': 44000,
		'LeftSideLabels': {'lt'},
		'RightSideLabels': {'rt'},
		'HighpassFilter': 'n/a',
		'LowpassFilter': 'n/a',
		'MERUnit': 'μV',
		'MERchannels':{
			'cen': 1,
			'ant': 2,
			'pos': 3,
			'med': 4,
			'lat': 5
		},
		'MERchannelsMap': {
			'cen': 'RAW 01 / Central',
			'ant': 'RAW 02 / Anterior',
			'pos': 'RAW 03 / Posterior',
			'med': 'RAW 04 / Medial',
			'lat': 'RAW 05 / Lateral'
		},
		'PowerLineFrequency': 60,
		'RecordingType': 'epoched',
		'ElectrodeInfo': {
			'Manufacturer': 'Alpha Omega',
			'Type': 'semi-micro',
			'Material': 'Tungsten',
			'Diameter': 100
		}
}

channelLabels = {1:"Center", 2:"Anterior", 3:"Posterior", 4:"Medial", 5:"Lateral"}

r = re.compile('.{4}-.{2}-.{2}T.{2}:.{2}:.{2}.{3}')

class WorkerKilledException(Exception):
	pass

class WorkerSignals(QtCore.QObject):
	'''
	Defines the signals available from a running worker thread.

	Supported signals are:

	finished
		No data
	
	error
		`tuple` (exctype, value, traceback.format_exc() )
	
	result
		`object` data returned from processing, anything

	progress
		`int` indicating % progress 

	'''

class mer2bids(QtCore.QRunnable):
	"""
	This class is a thread, which manages one thread of control within the GUI.
	
	:param new_sessions: Dictionary containing information about each subject. Wether there are new sessions to process in the input directory. 
	:type new_sessions: dictionary
	:param file_info: Information about each subject file in the input directory
	:type file_info: dictionary
	:param input_path: Path to the input directory.
	:type input_path: string
	:param output_path: path to the output directory
	:type output_path: string
	:param coordinates: Optional list of electrode coordinates (x,y,z)
	:type coordinates: list or None
	:param make_dir: Make the directory
	:type make_dir: boolean
	:param overwrite: If duplicate data is present in the output directory overwrite it.
	:type overwrite: boolean
	:param verbose: Print out process steps.
	:type verbose: boolean
	:param annotation_extract: Extract annotations from the edf file.
	:type annotation_extract: boolean
	:param compress: Compress the edf file
	:type compress: boolean
	
	"""
	
	def __init__(self):	
		super(mer2bids, self).__init__()
		
		self.input_path = []
		self.output_path = []
		self.script_path = []
		
# 		self.signals = WorkerSignals()
		
		self.running = False
		self.userAbort = False
		self.is_killed = False
		
	def stop(self):
		self.running = False
		self.userAbort = True
	
	def kill(self):
		self.is_killed = True
	
	def write_annotations(self, data_fname, annotation_fname):
		self.annotation_fname=annotation_fname
		self._annotations_data(data_fname)
	
	def microsoft_to_iso8601(self, microsoft):
		base_date = datetime.datetime(1899, 12, 31)
		ret_date = base_date + datetime.timedelta(days=microsoft)
		return ret_date.isoformat()
	
	def _read_bin(self,fname):
		with open(fname, mode='rb') as f:
			while True:
				data = f.read(4)
				if len(data) < 4:
					# end of file
					return
				yield struct.unpack('<2h', data)[0]
			
	def _write_header(self, header_path):
		
		if isinstance(header_path,str):
			data = pd.read_csv(header_path, sep="\t", header=0)
			data['time'] = [self.microsoft_to_iso8601(x) for x in data['RecordingTime']]
			
			data_tmp={'side_num':[],'side_label':[],'chan_num':[],'chan_label':[]}
			for irow in list(data['Description']):
				if ' L ' in irow.split(':')[-1].strip():
					side='left'
				elif ' R ' in irow.split(':')[-1].strip():
					side='right'
					
				data_tmp['side_num'].append(int(irow.split(':')[0].strip()))
				data_tmp['side_label'].append(str(side))
				data_tmp['chan_num'].append(int(irow.split(':')[2].strip().split(' ')[-1].strip()))
				data_tmp['chan_label'].append(str(irow.split(':')[1].strip().split(' ')[0].strip().lower()))
				
			data_tmp=pd.DataFrame(data_tmp)
			
			data=pd.concat([data, data_tmp], axis=1)
			data.sort_values(['time'], inplace=True)
			data.to_csv(os.path.splitext(header_path)[0]+'_new.txt', index = None, header=True, sep='\t',line_terminator="")
			
	def _get_data(self, systemInfo, header_path, isub):
		self.header={}
		if 'Medtronic' in systemInfo['Manufacturer']:
			head_data = pd.read_csv(header_path, sep="\t", header=0)
			for iside in head_data['side_label'].unique():
				meas_info_tmp={}
				meas_info_tmp['firstname']=isub.split('-')[0]
				meas_info_tmp['lastname']=isub.split('-')[1]
				meas_info_tmp['meas_date'] = min(head_data[head_data['side_label']==iside]['time'])
				meas_info_tmp['sampling_frequency'] = min(head_data[head_data['side_label']==iside]['SampleFreq'])
				meas_info_tmp['nchan']= head_data[head_data['side_label']==iside]['Channel'].unique()
				meas_info_tmp['equipment']=systemInfo['ManufacturersModelName']
				
				chan_info_tmp={}
				for ichan in sorted(head_data[head_data['side_label']==iside]['chan_num'].unique()):
					print('Importing file for side: {}, channel: {}'.format(iside, ichan))
					
					chan_data=head_data[(head_data['side_label']==iside) & (head_data['chan_num']==ichan)]
					
					chan_mer={'depth':[], 'date':[],'time':[],'data':[],'duration':[]}
					for ifile in list(chan_data['Filename']):
						test=[]
						for value in self._read_bin(os.path.join(os.path.dirname(header_path),ifile)):
							test.append(value/10)
						chan_mer['data'].append(test)
						chan_mer['depth'].append(float(head_data[head_data['Filename']==ifile]['Position'].values[0]/1000))
						chan_mer['date'].append(head_data[head_data['Filename']==ifile]['time'].values[0].split('T')[0])
						chan_mer['time'].append(head_data[head_data['Filename']==ifile]['time'].values[0].split('T')[-1])
						chan_mer['duration'].append(float(len(test)/int(chan_data['SampleFreq'].unique()[0])))
					
					chan_label = [i for i,x in enumerate(list(systemInfo['MERchannelsMap'])) if str(chan_data['chan_label'].unique()[0]) in x]
					if not chan_label:
						chan_label=str(chan_data['chan_label'].unique()[0])
					else:
						chan_label=list(systemInfo['MERchannelsMap'].values())[chan_label[0]]
						
					ch_dict = {
						'label': chan_label,
						'dimension': 'uV',
						'sample_rate': int(chan_data['SampleFreq'].unique()[0]),
						'physical_max': max(map(lambda x: x[3], chan_mer['data'])),
						'physical_min': min(map(lambda x: x[3], chan_mer['data'])),
						'digital_max': +32767,
						'digital_min': -32767,
						'transducer': '',
						'prefilter': ''
					}
					
					ch_dict={**ch_dict,**chan_mer}
					chan_info_tmp[ichan]=ch_dict
			
				self.header[iside]={}
				self.header[iside]['meas_info']=meas_info_tmp
				self.header[iside]['chan_info']=chan_info_tmp
			
		else:
			data_tmp={
				'filename':[],
				'sample_rate':[],
				'depth':[],
				'time':[],
				'side_label':[],
				'header':[],
				'file_data':[]
			}
			for ifile in header_path:
				print('Importing file: {}'.format(os.path.basename(ifile)))
				
				file_in=alphaOmegaIO(ifile)
				file_data,header=file_in.block_data(chansKeep={'RAW 01 / Central','RAW 02 / Anterior','RAW 03 / Posterior',
												  'RAW 04 / Medial','RAW 05 / Lateral'})
				data_tmp['filename'].append(ifile)
				data_tmp['sample_rate'].append(int(header['sampling_rate']))
				data_tmp['depth'].append(float(header['depth']))
				data_tmp['time'].append(header['datetime'])
				data_tmp['side_label'].append(header['side'])
				data_tmp['header'].append(header)
				data_tmp['file_data'].append(file_data)
			
			data_tmp=pd.DataFrame(data_tmp)
			side_order=[]
			for iside in data_tmp['side_label'].unique():
				side_data=data_tmp[data_tmp['side_label']==iside].reset_index(drop=True)
				meas_info_tmp={}
				meas_info_tmp['firstname']=isub.split('-')[0]
				meas_info_tmp['lastname']=isub.split('-')[1]
				meas_info_tmp['meas_date'] = min(side_data[side_data['side_label']==iside]['time'])
				meas_info_tmp['sampling_frequency'] = min(side_data[side_data['side_label']==iside]['sample_rate'])
				meas_info_tmp['nchan']= len(side_data[side_data['side_label']==iside]['file_data'][0]['name'])
				meas_info_tmp['equipment']=systemInfo['ManufacturersModelName']
				
				chan_info_tmp={}
				for ichan in range(meas_info_tmp['nchan']):
					chan_mer={'depth':[], 'date':[],'time':[],'data':[],'duration':[]}
					for ifile in range(side_data[side_data['side_label']==iside].shape[0]):
						chan_mer['data'].append(side_data[side_data['side_label']==iside]['file_data'][ifile]['data'][ichan])
						chan_mer['depth'].append(side_data[side_data['side_label']==iside]['header'][ifile]['depth'])
						chan_mer['date'].append(side_data[side_data['side_label']==iside]['header'][ifile]['datetime'].split('T')[0])
						chan_mer['time'].append(side_data[side_data['side_label']==iside]['header'][ifile]['datetime'].split('T')[-1])
						chan_mer['duration'].append(float(len(side_data[side_data['side_label']==iside]['file_data'][ifile]['data'][ichan])/meas_info_tmp['sampling_frequency']))
					
					chan_label=side_data[side_data['side_label']==iside]['file_data'][0]['name'][ichan]
					chan_num=int(''.join([x for x in chan_label if x.isdigit()]))
					final_label=[]
					for label, systemLabel in systemInfo['MERchannelsMap'].items():
						if chan_label in systemLabel:
							final_label=label
					if not final_label:
						final_label=chan_label
						
					ch_dict = {'label': final_label,
							'dimension': 'uV',
							'sample_rate': meas_info_tmp['sampling_frequency'],
							'physical_max': max(map(lambda x: x[3], chan_mer['data'])),
							'physical_min': min(map(lambda x: x[3], chan_mer['data'])),
							'digital_max': +32767,
							'digital_min': -32767,
							'transducer': '',
							'prefilter': ''}
					
					ch_dict={**ch_dict,**chan_mer}
					sort_index = np.argsort([str(a) +'T' +b for a,b in zip(ch_dict['date'],ch_dict['time'])])
					ch_dict['data']=[ch_dict['data'][i] for i in sort_index]
					ch_dict['date']=[ch_dict['date'][i] for i in sort_index]
					ch_dict['depth']=[ch_dict['depth'][i] for i in sort_index]
					ch_dict['time']=[ch_dict['time'][i] for i in sort_index]
					ch_dict['duration']=[ch_dict['duration'][i] for i in sort_index]
					
					chan_info_tmp[chan_num]=ch_dict
				
				side_order.append([iside,meas_info_tmp['meas_date']])
				
				self.header[iside]={}
				self.header[iside]['meas_info']=meas_info_tmp
				self.header[iside]['chan_info']=chan_info_tmp
				
			sorted_idx=np.argsort([x[1] for x in side_order])
			sorted_head={}
			for iidx in sorted_idx:
				sorted_head[side_order[iidx][0]] = self.header[side_order[iidx][0]]
			
			self.header=sorted_head
		
		return self.header
		
	def _read_annotations_apply_offset(self, triggers):
		events = []
		offset = 0.
		for k, ev in enumerate(triggers):
			onset = float(ev[0]) + offset
			duration = float(ev[2]) if ev[2] else 0
			for description in ev[3].split('\x14')[1:]:
				if description:
					events.append([onset, duration, description, ev[4]])
				elif k==0:
					# "The startdate/time of a file is specified in the EDF+ header
					# fields 'startdate of recording' and 'starttime of recording'.
					# These fields must indicate the absolute second in which the
					# start of the first data record falls. So, the first TAL in
					# the first data record always starts with +0.X2020, indicating
					# that the first data record starts a fraction, X, of a second
					# after the startdate/time that is specified in the EDF+
					# header. If X=0, then the .X may be omitted."
					offset = -onset
					
		return events if events else list()
	
	def _read_annotation_block(self, data_fname, block, tal_indx):
		pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
		assert(block>=0)
		data = []
		with open(data_fname, 'rb') as fid:
			assert(fid.tell() == 0)
			blocksize = np.sum(self.header['chan_info']['n_samps']) * self.header['meas_info']['data_size']
			fid.seek(np.int64(self.header['meas_info']['data_offset']) + np.int64(block) * np.int64(blocksize))
			read_idx = 0
			for i in range(self.header['meas_info']['nchan']):
				read_idx += np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size'])
				buf = fid.read(np.int64(self.header['chan_info']['n_samps'][i]*self.header['meas_info']['data_size']))
				if i==tal_indx:
					raw = re.findall(pat, buf.decode('latin-1'))
					if raw:
						data.append(list(map(list, [x+(block,) for x in raw])))
			
		return data
	
	def _annotations_data(self, data_fname):
		"""
		Constructs an annotations data tsv file about patient specific events from edf file.
		
		:param file_info_run: File header information for specific recording.
		:type file_info_run: dictionary
		:param annotation_fname: Filename for the annotations tsv file.
		:type annotation_fname: string
		:param data_fname: Path to the raw data file for specific recording.
		:type data_fname: string
		:param overwrite: If duplicate data is present in the output directory overwrite it.
		:type overwrite: boolean
		:param verbose: Print out process steps.
		:type verbose: boolean
		
		"""
		
		file_in = EDFReader()
		file_in.open(data_fname)
		self.header = file_in.readHeader()
		
		tal_indx = [i for i,x in enumerate(self.header['chan_info']['ch_names']) if x.endswith('Annotations')][0]
		
		start_time = 0
		end_time = self.header['meas_info']['n_records']*self.header['meas_info']['record_length']
		
		begsample = int(self.header['meas_info']['sampling_frequency']*float(start_time))
		endsample = int(self.header['meas_info']['sampling_frequency']*float(end_time))
		
		n_samps = max(set(list(self.header['chan_info']['n_samps'])), key = list(self.header['chan_info']['n_samps']).count)
		
		begblock = int(np.floor((begsample) / n_samps))
		endblock = int(np.floor((endsample) / n_samps))
		
		update_cnt = int((endblock+1)/10)
		annotations = []
		for block in range(begblock, endblock):
			if self.is_killed:
				self.running = False
				raise WorkerKilledException
			else:
				data_temp = self._read_annotation_block(data_fname, block, tal_indx)
				if data_temp:
						annotations.append(data_temp[0])
# 				if block == update_cnt and block < (endblock-(int((endblock+1)/20))):
# 					print('{}%'.format(int(np.ceil((update_cnt/endblock)*100))))
# 					update_cnt += int((endblock+1)/10)
		
		events = self._read_annotations_apply_offset([item for sublist in annotations for item in sublist])
		
		annotation_data = pd.DataFrame({})
		if events:
			fulldate = datetime.datetime.strptime(self.header['meas_info']['meas_date'], '%Y-%m-%d %H:%M:%S')
			for i, annot in enumerate(events):
				data_temp = {'onset': annot[0],
							 'duration': annot[1],
							 'time_abs': (fulldate + datetime.timedelta(seconds=annot[0]+float(self.header['meas_info']['millisecond']))).strftime('%H:%M:%S.%f'),
							 'time_rel': sec2time(annot[0], 3),
							 'event': annot[2]}
				annotation_data = pd.concat([annotation_data, pd.DataFrame([data_temp])], axis = 0)
			
		annotation_data.to_csv(self.annotation_fname, sep='\t', index=False, na_rep='n/a', line_terminator="", float_format='%.3f')
	
	def _write_edf_file(self,data_fname, patient_data, irun):
		
		ms_info=patient_data[irun]['meas_info']
		ch_info=patient_data[irun]['chan_info']
# 		ms_info=patient_data['lt2']['meas_info']
# 		ch_info=patient_data['lt2']['chan_info']
		
		file_out = EDFwriter(data_fname, EDFwriter.EDFLIB_FILETYPE_EDFPLUS, len(ch_info))
		
		start_datetime = datetime.datetime.strptime(ms_info['meas_date'], '%Y-%m-%dT%H:%M:%S.%f')
		file_out.setStartDateTime(start_datetime.year,start_datetime.month, start_datetime.day,start_datetime.hour,start_datetime.minute,start_datetime.second,start_datetime.microsecond/100)
		
# 		if file_out.setPatientCode('X X X X') != 0:
# 			print("setPatientCode() returned an error")
# 		if file_out.setPatientBirthDate(1913, 4, 7) != 0:
# 			print("setPatientBirthDate() returned an error")
		if file_out.setPatientName(','.join([ms_info['lastname'],ms_info['firstname']])) != 0:
			print("setPatientName() returned an error")
# 		if file_out.setAdditionalPatientInfo("normal condition") != 0:
# 			print("setAdditionalPatientInfo() returned an error")
		if file_out.setAdministrationCode(irun) != 0:
			print("setAdministrationCode() returned an error")
# 		if file_out.setTechnician("Black Jack") != 0:
# 			print("setTechnician() returned an error")
		if file_out.setEquipment(ms_info['equipment']) != 0:
			print("setEquipment() returned an error")
# 		if file_out.setAdditionalRecordingInfo("nothing special") != 0:
# 			print("setAdditionalRecordingInfo() returned an error")

		ch_cnt=0
		for ichan in list(ch_info):
			if file_out.setPhysicalMaximum(ch_cnt, ch_info[ichan]['physical_max'])!= 0:
				print("setPhysicalMaximum() for chan {} returned an error".format(ichan))
			if file_out.setPhysicalMinimum(ch_cnt,ch_info[ichan]['physical_min'])!= 0:
				print("setPhysicalMinimum() for chan {} returned an error".format(ichan))
			if file_out.setDigitalMaximum(ch_cnt, ch_info[ichan]['digital_max'])!= 0:
				print("setDigitalMaximum() for chan {} returned an error".format(ichan))
			if file_out.setDigitalMinimum(ch_cnt, ch_info[ichan]['digital_min'])!= 0:
				print("setDigitalMinimum() for chan {} returned an error".format(ichan))
			if file_out.setPhysicalDimension(ch_cnt, ch_info[ichan]['dimension'])!= 0:
				print("setPhysicalDimension() for chan {} returned an error".format(ichan))
			if file_out.setSignalLabel(ch_cnt, ch_info[ichan]['label'])!= 0:
				print("setSignalLabel() for chan {} returned an error".format(ichan))
			if file_out.setPreFilter(ch_cnt,  ch_info[ichan]['prefilter'])!= 0:
				print("setPreFilter() for chan {} returned an error".format(ichan))
			if file_out.setTransducer(ch_cnt, ch_info[ichan]['transducer'])!= 0:
				print("setTransducer() for chan {} returned an error".format(ichan))
			if file_out.setSampleFrequency(ch_cnt, ch_info[ichan]['sample_rate'])!= 0:
				print("setSampleFrequency() for chan {} returned an error".format(ichan))
				
			ch_cnt+=1
		
		maxIdx,maxList=max(enumerate([ch_info[x]['depth'] for x in list(ch_info.keys())]), key = lambda x: len(x[1]))
		
		for idepth in maxList:
			chansPresent=[x for x in list(ch_info.keys()) if idepth in ch_info[x]['depth']]
			depthCnt={}
			for ipresent in chansPresent:
				depthCnt[ipresent]=[i for i,x in enumerate(ch_info[ipresent]['depth']) if idepth == x][0]
			
			minLen=min(map(lambda x: len(x), [ch_info[x]['data'][depthCnt[x]] for x in chansPresent]))
			minSF=min(map(lambda x: x, [ch_info[x]['sample_rate'] for x in chansPresent]))
			
			for irec in range(0, int(minLen/minSF)):
				for ichan in list(ch_info):
					if idepth not in ch_info[ichan]['depth']:
						buf=np.zeros(ch_info[ichan]['sample_rate'], dtype=np.int32,order='C')
					else:
						buf = np.array(ch_info[ichan]['data'][depthCnt[ichan]][irec*ch_info[ichan]['sample_rate']:((irec+1)*ch_info[ichan]['sample_rate'])+1], dtype=np.int32,order='C')
					if file_out.writeSamples(buf) != 0:
						print("writeSamples() returned an error on:")
		
		onset=0
		for idepth in range(len(maxList)):
			err=file_out.writeAnnotation(onset, 0, 'depth: {}'.format(maxList[idepth]))
			if err != 0:
				print("writeAnnotation() returned an error on:")
				
			onset+= (int(len(ch_info[list(ch_info.keys())[maxIdx]]['data'][idepth])/ch_info[list(ch_info.keys())[maxIdx]]['sample_rate'])*10000)
			
		file_out.close()
			
	def _get_run_info(self, data_fname, side):
			file_in = EDFReader()
			header = file_in.open(data_fname)
			
			file_info = [
					('FileName', data_fname),
					('Subject', header['meas_info']['subject_id']),
					('Gender', header['meas_info']['gender']),
					('Age', 'X'),
					('Birthdate', header['meas_info']['birthdate']),
					('RecordingID', header['meas_info']['recording_id']),
					('Date', header['meas_info']['meas_date'].split(' ')[0]),
					('Time', header['meas_info']['meas_date'].split(' ')[1]),
					('DataOffset', header['meas_info']['data_offset']),
					('NRecords', header['meas_info']['n_records']),
					('RecordLength', header['meas_info']['record_length']),
					('TotalRecordTime', round((((header['meas_info']['n_records']*(header['meas_info']['sampling_frequency']*header['meas_info']['record_length']))/header['meas_info']['sampling_frequency'])/60)/60,3)),
					('NChan', header['meas_info']['nchan']),
					('SamplingFrequency', header['meas_info']['sampling_frequency']),
					('Highpass', header['meas_info']['highpass']),
					('Lowpass', header['meas_info']['lowpass']),
					('Groups', side),
					('EDF_type', header['meas_info']['subtype'])]
			
			file_info = OrderedDict(file_info)
				
			file_info['RecordingType'] = 'MER'
			file_info['RecordingLength'] = 'Full'
			file_info['Retro_Pro'] = 'Pro'
			chan_type = 'MER'
			
			chan_idx = [i for i, x in enumerate(header['chan_info']['ch_names']) if x !='EDF Annotations']

			ch_info = {}
			ch_info[chan_type] = OrderedDict([
				('ChannelCount', len([header['chan_info']['ch_names'][x] for x in chan_idx])),
				('Unit', np.array(header['chan_info']['units'])[chan_idx]),
				('ChanName', np.array(header['chan_info']['ch_names'])[chan_idx]),
				('Type', chan_type)])
			
			file_info['ChanInfo'] = OrderedDict(ch_info)
			file_info = OrderedDict(file_info)
		
			return file_info



# 	m2b=mer2bids()
	@QtCore.Slot()
	def run(self):
		"""
		Main loop for building BIDS database.
				
		"""
		self.running = True
		
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path)
			
		folders = sorted_nicely([x for x in os.listdir(self.input_path) if os.path.isdir(os.path.join(self.input_path, x)) and x != 'raw'])
# 		folders = sorted_nicely([x for x in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, x)) and x != 'raw'])
		
		participants_fname = bidsHelper(output_path=self.output_path).write_participants(return_fname=True)
		if not os.path.exists(participants_fname):
			participants_fname = bidsHelper(output_path=self.output_path).write_participants()
		
		self.participant_tsv = pd.read_csv(participants_fname, sep='\t')
		
		while self.running:
			try:
				for isub in folders:
					if isub not in list(self.participant_tsv['participant_id']):
						sub_fold = os.path.join(self.input_path,isub)
		# 				sub_fold = os.path.join(input_path,isub)
						sessions = [x for x in os.listdir(sub_fold) if x.endswith('header.txt')]
						if not sessions:
							sessions = [x for x in os.listdir(sub_fold) if os.path.isdir(os.path.join(sub_fold,x))]
							if not sessions:
								sessions=[0]
							
						for ihead in sessions:
							#--- Check for .mat files, true if Alpha Omega
							file_list = [x.lower() for x in os.listdir(sub_fold) if x.endswith('.mpx')]
							if len(file_list) > 0:
								systemInfo = alphaomega_info
								file_list=[os.path.join(sub_fold,x) for x in file_list]
								patient_data = self._get_data(systemInfo, file_list, isub)
		# 						patient_data = m2b._get_data(systemInfo, file_list, isub)
							else:
								systemInfo=leadpoint_info
								file_list = [x for x in os.listdir(sub_fold) if x.endswith('.bin') and 'header' not in x]
								if not os.path.exists(os.path.join(sub_fold,os.path.splitext(ihead)[0]+'_new.txt')):
									self._write_header(os.path.join(sub_fold,ihead))
								
								patient_data = self._get_data(systemInfo, os.path.join(sub_fold,os.path.splitext(ihead)[0]+'_new.txt'), isub)
		# 						patient_data = m2b._get_data(systemInfo, os.path.join(sub_fold,os.path.splitext(ihead)[0]+'_new.txt'), isub)
							
							bids_helper = bidsHelper(subject_id=isub, session_id='perisurg', kind='ieeg', output_path=self.output_path, systemInfo=systemInfo, json_metadata=mer_file_metadata,make_sub_dir=True)
		# 					bids_helper = bidsHelper(subject_id=isub, session_id='perisurg', kind='ieeg', output_path=output_path, systemInfo=systemInfo, json_metadata=mer_file_metadata,make_sub_dir=True)
							
							info_run={}
							run_cnt=0
							for irun in list(patient_data):
								print('Starting: {} run {}'.format(isub,irun))
								run_num = str(run_cnt+1).zfill(2)
								bids_helper.run_num=run_num
								run_cnt+=1
								
								data_fname = bids_helper.make_bids_filename(suffix = 'ieeg' + '.edf')
								if not os.path.exists(data_fname):
									
									annotation_fname = bids_helper.make_bids_filename(suffix='annotations.tsv')
									
									self._write_edf_file(data_fname, patient_data, irun)
									self.write_annotations(data_fname, annotation_fname)

# 									m2b._write_edf_file(data_fname, patient_data, irun)
# 									m2b.write_annotations(data_fname, annotation_fname)
									
									info_run[irun]=self._get_run_info(data_fname, irun)
									
# 									info_run[irun]=m2b._get_run_info(data_fname, irun)
			
									scan_fname=data_fname.split('perisurg'+os.path.sep)[-1].replace(os.path.sep,'/')
									bids_helper.write_scans(scan_fname, info_run[irun], False)
									
									bids_helper.write_channels(info_run[irun])
									bids_helper.write_sidecar(info_run[irun])
									
									print('Finished: {} run {}\n'.format(isub,irun))
								else:
									print('Already exists: {} run {}\n'.format(isub,irun))
							
							if info_run:
								bids_helper.write_electrodes(info_run, coordinates=None)
							
							code_output_path = os.path.join(self.output_path, 'code', 'mer2bids')
							code_path = bids_helper.make_bids_folders(path_override=code_output_path, make_dir=False)
							if not os.path.exists(os.path.join(code_path,'mer2bids.py')):
								code_path = bids_helper.make_bids_folders(path_override=code_output_path, make_dir=True)
								shutil.copy(os.path.join(self.script_path, 'mer2bids.py'), code_path)
							
						info_sub={}
						info_sub['Age']=None
						info_sub['Gender']=None
						bids_helper.write_participants([info_sub])
							
						self.participant_tsv = pd.read_csv(participants_fname, sep='\t')
								
					else:
						print('Participant {} already exists in the dataset!'.format(isub))
					
			except WorkerKilledException:
				self.running = False
				print('Except!')
				pass
			
			finally:
				self.running = False
				print('Finished.')

#%%

input_path = r'/media/veracrypt6/projects/mer_analysis/mer/source'
output_path = r'/media/veracrypt6/projects/mer_analysis/working_dir'
script_path = r'/home/greydon/Documents/GitHub/python_modules/mer2bids'

# Set Qthread
worker = mer2bids()
threadpool = QtCore.QThreadPool()

worker.input_path = r'/media/veracrypt6/projects/mer_analysis/mer/source'
worker.output_path = r'/media/veracrypt6/projects/mer_analysis/working_dir/out'
worker.script_path = r'/home/greydon/Documents/GitHub/python_modules/mer2bids'

# Execute

threadpool.start(worker)




