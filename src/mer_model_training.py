#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 21:20:52 2021

@author: greydon
"""

import os
from bids import BIDSLayout
import pyedflib
import numpy as np
import re
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn import model_selection
from sklearn.pipeline import Pipeline
import glob
import pickle
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pytablewriter import MarkdownTableWriter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve,roc_auc_score
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mtick
os.chdir('/home/greydon/Documents/GitHub/merPrep/src')
import merPrep as mer
import matplotlib.patches as mpatches


channelLabels = {'cen':"Center", 'ant':"Anterior", 'pos':"Posterior", 'med':"Medial", 'lat':"Lateral"}


def compute_features(chan_labels, sigbufs, annots, sf,ichan):
	index=[i for i,x in enumerate(chan_labels) if x == ichan][0]
	depths=[float(re.findall(r"[-+]?\d*\.\d+|\d+",x)[0]) for x in annots[2]]
	sampling_f=sf[index]
	signal_features=[]
	for idepth in range(len(annots[2])):
		start=int(abs(annots[0])[idepth]*sf[index])
		if idepth < (len(annots[2])-1):
			end=int(annots[0][idepth+1]*sf[index])
		else:
			end=n_samps
		
		tempData = sigbufs[index,start:end]
		
		if idepth < (len(annots[2])-3):
			startMAV=int(annots[0][idepth+1]*sf[index])
			endMAV=int(annots[0][idepth+2]*sf[index])
			MAVtemp = mer.MAVS(tempData, sigbufs[index,startMAV:endMAV])
		else:
			MAVtemp = np.nan
		
		signal_features.append([
			isubject, 
			side_label,
			ichan,
			depths[idepth],
			mer.MAV(tempData), 
			MAVtemp,
			mer.VAR(tempData), 
			mer.MMAV1(tempData), 
			mer.MMAV2(tempData), 
			mer.RMS(tempData), 
			mer.curveLen(tempData), 
			mer.zeroCross(tempData,10), 
			mer.threshold(tempData), 
			mer.WAMP(tempData,10), 
			mer.SSI(tempData), 
			mer.powerAVG(tempData), 
			mer.peaksNegPos(tempData), 
			mer.tkeoTwo(tempData), 
			mer.tkeoFour(tempData),
			mer.shapeFactor(tempData), 
			mer.KUR(tempData), 
			mer.SKW(tempData), 
			mer.meanFrq(tempData,sampling_f),
			mer.powerAVG(mer.butterBandpass(tempData, lowcut = 500, highcut = 3000, fs = sampling_f, order = 5)),
			mer.entropy(tempData),
			mer.wavlet(tempData, nLevels = 5, waveletName = 'db1', timewindow = False, windowSize = 0, Fs=sf)
			])
		
		print(f"Finished channel {ichan} depth {depths[idepth]}")
		
	return signal_features

def clean_dataset(df):
	assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
	df.dropna(inplace=True)
	indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
	return df[indices_to_keep].astype(np.float64)

def remove_correlated_features(X, corr_threshold = 0.9):
	y=None
	sub=None
	if 'Class' in list(X):
		y = pd.DataFrame(X['Class'].astype(int))
		X = X.drop(['Class'], axis=1)
	
	if 'subject' in list(X):
		sub = pd.DataFrame(X['subject'].astype(int))
		X = X.drop(['subject'], axis=1)
	
	corr = X.corr()
	drop_columns = np.full(corr.shape[0], False, dtype=bool)
	for i in range(corr.shape[0]):
		for j in range(i + 1, corr.shape[0]):
			if abs(corr.iloc[i, j]) >= corr_threshold:
				drop_columns[j] = True
	columns_dropped = X.columns[drop_columns]
	X.drop(columns_dropped, axis=1, inplace=True)
	if y is not None:
		X.insert(0,'Class',y)
	if sub is not None:
		X.insert(0,'subject',sub)
	print(f"Columns dropped: {columns_dropped.values}")
	return X.reset_index(drop=True)

def backward_elimination(X, threshold = 0.1):
	y=None
	sub=None
	if 'Class' in list(X):
		y = pd.DataFrame(X['Class'].astype(int))
		X = X.drop(['Class'], axis=1)
	
	if 'subject' in list(X):
		sub = pd.DataFrame(X['subject'].astype(int))
		X = X.drop(['subject'], axis=1)
		
	if y is not None:
		# adding constant column of ones, mandatory for sm.OLS model
		X_1 = sm.add_constant(X) # fitting sm.OLS model
		model = sm.OLS(y, X_1).fit()
		# backward elimination
		cols = list(X.columns)
		pmax = 1
		while (len(cols)>0):
			p = []
			X_1 = X[cols]
			X_1 = sm.add_constant(X_1)
			model = sm.OLS(y, X_1).fit()
			p = pd.Series(model.pvalues.values[1:],index = cols)
			pmax = max(p)
			feature_with_p_max = p.idxmax()
			if(pmax > threshold):
				cols.remove(feature_with_p_max)
			else:
				break
		selected_features_BE = cols
		X = X[selected_features_BE]
		X.insert(0,'Class',y)
		if sub is not None:
			X.insert(0,'subject',sub)
		print(f"Columns kept: {selected_features_BE}")
	return X.reset_index(drop=True)

from sklearn.base import clone 

def imp_df(column_names, importances):
	data = {
		'Feature': column_names,
		'Importance': importances,
	}
	df = pd.DataFrame(data).set_index('Feature').sort_values('Importance', ascending=False)
	return df

def drop_col_feat_imp(model, X_train, y_train, X_test, y_test, random_state = 42):
	# clone the model to have the exact same specification as the one initially trained
	model_clone = clone(model)
	# set random_state for comparability
	model_clone.random_state = random_state
	# training and scoring the benchmark model
	model_clone.fit(X_train, y_train)
	y_pred=model_clone.predict(X_test)
	benchmark_score=metrics.accuracy_score(y_test, y_pred)

	# list for storing feature importances
	importances = []
	
	# iterating over all columns and storing feature importance (difference between benchmark and new model)
	for col in X_train.columns:
		model_clone = clone(model)
		model_clone.random_state = random_state
		model_clone.fit(X_train.drop(col, axis = 1), y_train)
		y_pred=model_clone.predict(X_test.drop(col, axis = 1))
		drop_col_score=metrics.accuracy_score(y_test, y_pred)

		importances.append(benchmark_score - drop_col_score)
	
	importances_df = imp_df(X_train.columns, importances)
	return importances_df

def outlier_removal_IQR(data):
	Q1=data.quantile(0.25)
	Q3=data.quantile(0.75)
	iqr=20*(Q3-Q1)
	for feature in [x for x in list(iqr.index) if x.lower() != 'class']:
		q1_idx = data[data[feature] < Q1[feature]-iqr[feature]].index
		data = data.drop(q1_idx)
		q3_idx = data[data[feature] > Q3[feature]+iqr[feature]].index
		data = data.drop(q3_idx)
	
	return data

feature_label_dict={
	'variance':'variance',
	'rms':'root mean square',
	'curveLength':'curve length',
	'zeroCross':'zero crossings',
	'threshold':'threshold',
	'wamp':'willison amplitude',
	'ssi':'simple square integral',
	'peaks':'peaks',
	'tkeoTwo':'Teager-Kaiser energy 2',
	'tkeoFour':'Teager-Kaiser energy 4',
	'shapeF':'shape factor',
	'skew':'skewness',
	'meanF':'mean frequency',
	'AvgPowerMU': 'multi-unit power',
	'waveletStd':'wavelet transform',
	'mmav2':'Modified mean abs val 2',
	'mmav1':'Modified mean abs val 1',
	'mav':'Mean abs val',
	'mavSlope':'Mean abs val slope',
	'kurtosis':'Kurtosis',
	'entropy':'Entropy',
	'power':'Power'
}


#%%

bids_dir=r'/media/veracrypt6/projects/mer_analysis/mer/bids'
out_path='/media/veracrypt6/projects/mer_analysis/mer/deriv/imgs'

xls = pd.ExcelFile('/media/veracrypt6/projects/stealthMRI/resources/excelFiles/patient_info.xlsx')
surgical_data=xls.parse('surgical', header=3)

layout = BIDSLayout(bids_dir)


#%%


cols=['subject','side','chan','depth','mav','mavSlope','variance','mmav1','mmav2','rms','curveLength','zeroCross','threshold','wamp','ssi',
				 'power','peaks','tkeoTwo','tkeoFour','shapeF','kurtosis','skew','meanF','AvgPowerMU','entropy','waveletStd']

ignore_subs=['P061']

for isubject in layout.get_subjects()[::-1][9:]:
	if isubject not in ignore_subs:
		edf_files=layout.get(subject=isubject, extension='.edf', return_type='filename')
		for iedf in edf_files:
			outname=os.path.join(iedf.split(f'/bids/sub-{isubject}')[0],'deriv','features2',f'sub-{isubject}',os.path.splitext(os.path.basename(iedf))[0].replace('ieeg','features')+'.pkl')
			if not os.path.exists(outname):
				if not os.path.exists(os.path.dirname(outname)):
					os.makedirs(os.path.dirname(outname))
				
				f = pyedflib.EdfReader(iedf)
				annots=f.readAnnotations()
				n = f.signals_in_file
				n_samps=f.getNSamples()[0]
				sf=f.getSampleFrequencies()
				side_label=f.admincode.decode('latin-1')
				sigbufs = np.zeros((n,n_samps))
				chan_labels=[]
				for i in np.arange(n):
					chan_labels.append(f.signal_label(i).decode('latin-1').strip())
					sigbufs[i, :] = mer.butterBandpass(f.readSignal(i), lowcut = 550, highcut = 4500, fs = sf[i], order = 3)
					
				f.close()
				
				signal_features=[]
				pool = Pool(3)
				func = partial(compute_features, chan_labels,sigbufs,annots,sf)
				
				for result in pool.imap(func, chan_labels):
					signal_features.append(result)
				
				df1 = pd.DataFrame(np.vstack(signal_features), columns=cols)
				df1.to_pickle(outname)


#%%

downsampleFactor = 5


for isubject in layout.get_subjects()[::-1][9:]:
	isub=int(''.join([x for x in isubject if x.isnumeric()]))
	sub_data=surgical_data[surgical_data['subjectNumber']==isub]
	if sub_data['target'].values[0].lower() =='stn':
		edf_files = glob.glob(os.path.join(bids_dir,f'sub-{isubject}','ses-perisurg','ieeg','*_ieeg.edf'))
		for iedf in edf_files:
			f = pyedflib.EdfReader(iedf)
			annots=f.readAnnotations()
			num_chans = f.signals_in_file
			n_samps = f.getNSamples()[0]
			sf = f.getSampleFrequencies()/downsampleFactor
			side_label=f.admincode.decode('latin-1')
			sigbufs = np.zeros((num_chans, int(n_samps/downsampleFactor)))
			chan_labels=[]
			for i in np.arange(num_chans):
				chan_labels.append(f.signal_label(i).decode('latin-1').strip())
				
				tempdat = mer.butterBandpass(f.readSignal(i), lowcut = 700, highcut =3000, fs = sf[i]*downsampleFactor, order = 3)
				tempdat = mer.downsample(tempdat, sf[i]*downsampleFactor,sf[i])
				
				sigbufs[i, :] = tempdat
			
			f.close()
			
			depths=[float(re.findall(r"[-+]?\d*\.\d+|\d+",x)[0]) for x in annots[2]]
			epoch_lens=[int(x*(sf[0])) for x in annots[0]]
			
			iside=None
			if any(x in side_label for x in ('lt','25','19','left')):
				iside='Left'
			elif any(x in side_label for x in ('rt','26','20','right')):
				iside='Right'
			
			if iside is not None:
				for ichan in range(num_chans):
					fileName=f"sub-{isubject}_side-{iside.lower()}_channel-{channelLabels[chan_labels[ichan]].lower()}_raw"
					if not os.path.exists(os.path.join(out_path,fileName+".png")):
						dorsal=sub_data[f'Surg{iside[0].title()[0]}{chan_labels[ichan].title()}In'].values[0]
						ventral=sub_data[f'Surg{iside[0].title()[0]}{chan_labels[ichan].title()}Out'].values[0]
						
						feature=[]
						for idepth in range(len(depths)):
							if idepth != len(depths)-1:
								tempdat = sigbufs[ichan,epoch_lens[idepth]:epoch_lens[idepth+1]]
							else:
								tempdat = sigbufs[ichan,epoch_lens[idepth]:]
							
							feature.append(tempdat)
						
						min_len = len(min(feature, key=len))
						max_len = len(max(feature, key=len))
						feature=np.vstack([x[:min_len] for x in feature])
						
						nDepths = len(depths)
						if sigbufs[ichan,:].max()-sigbufs[ichan,:].min() > 80:
 							yshift = round((sigbufs[ichan,:].max()-sigbufs[ichan,:].min())/100,1)
						else:
 							yshift = round((sigbufs[ichan,:].max()-sigbufs[ichan,:].min())/10,1)

						#yshift= int((sigbufs[ichan,:].max()-sigbufs[ichan,:].min())/2)
						
						#plt.ioff()
						#plt.ion()
						fig, ax = plt.subplots(figsize=(14,10))
						ax.plot(feature.T + yshift * np.arange(0, nDepths, 1), color='black', linewidth=0.2)
						#matplotlib.use('Qt5Agg')
						ax.yaxis.set_ticks(yshift * np.arange(0, nDepths, 1))
						ax.yaxis.set_ticklabels(['{:.2f}'.format(x) for x in depths])
						ax.xaxis.set_ticks(np.arange(0,len(feature.T)+1,(len(feature.T)/5)))
						start, end = ax.get_xlim()
						xTickLabs = np.arange(0, len(feature.T)+1, int(len(feature.T)/5))/sf[ichan]
						ax.xaxis.set_ticklabels(['{:.2f}'.format(x) for x in xTickLabs])
						ax.set_ylim(-yshift,(nDepths*yshift))
						ax.set_xlim(0,len(feature.T))
						
						ax.set_xlabel('Time (sec)', fontsize=18, fontweight='bold',labelpad=14)
						ax.set_ylabel('Depth (mm)', fontsize=18, fontweight='bold')
						
						ax.spines['right'].set_visible(False)
						ax.spines['top'].set_visible(False)
						
						ax.tick_params(axis='both', which='major', labelsize=14)
						handles=[]
						if not np.isnan(dorsal) and not np.isnan(ventral):
							idx_dor,val_dor = min(enumerate(depths), key=lambda x: abs(x[1]-dorsal))
							idx_ven,val_ven = min(enumerate(depths), key=lambda x: abs(x[1]-ventral))
							
							ax.axhline(idx_dor*yshift, color='#4daf4a', linewidth=2,zorder=11,label='Dorsal Border')
							ax.axhline(idx_ven*yshift, color='#e41a1c', linewidth=2,zorder=11,label='Ventral Border')
							ax.add_patch(mpatches.Rectangle((0, idx_dor*yshift), len(feature.T), ((idx_ven-idx_dor))*yshift, alpha=.3, facecolor='#0868ac', zorder=10, label='STN'))
							
							#ax.add_patch(mpatches.Rectangle((0, -yshift), len(feature.T), (idx_dor+1)*yshift, alpha=.4, facecolor='#0868ac', zorder=10))
							#ax.add_patch(mpatches.Rectangle((0, idx_ven*yshift), len(feature.T), ((nDepths-idx_ven)+1)*yshift, alpha=.4, facecolor='#0868ac', zorder=10))
							#ax.add_patch(mpatches.Rectangle((0, idx_dor*yshift), len(feature.T), ((idx_ven-idx_dor))*yshift, alpha=.4, facecolor='#dede00', zorder=10))
							
							handles, labels = ax.get_legend_handles_labels()
							plt.legend(bbox_to_anchor= (1.02, .6), shadow = True, facecolor = 'white',handles=handles,fontsize=16)
						
						plt.gca().invert_yaxis()
						plt.tight_layout()
						plt.subplots_adjust(right=0.8,top=0.95)
						
						plt.title(f"sub-{isubject} {iside} Side: {channelLabels[chan_labels[ichan]]} Trajectory", size=24, fontweight="bold",y=1.01)
						#plt.show()
						fileName=f"sub-{isubject}_side-{iside.lower()}_channel-{channelLabels[chan_labels[ichan]].lower()}_raw"
						plt.savefig(os.path.join(out_path,f"{fileName}.svg"),transparent=True,dpi=500)
						plt.savefig(os.path.join(out_path,f"{fileName}.png"),transparent=True,dpi=500)
						plt.savefig(os.path.join(out_path,f"{fileName}_white.png"),transparent=False,dpi=500)
						plt.close()
						
						print(f'Finished {isubject} {iside.lower()} side {channelLabels[chan_labels[ichan]].lower()} channel')
						
						del feature,handles
					
			del sigbufs

#							out_patch = mpatches.Patch(color='#0868ac', label='Label 0')
#							handles.append(out_patch)
#							in_patch = mpatches.Patch(color='#dede00', label='Label 1')
#							handles.append(in_patch)
#%%
class ScalarFormatterForceFormat(mtick.ScalarFormatter):
	def _set_format(self):  # Override function that finds format to use.
		self.format = "%1.1f"  # Give format here

out_path='/media/greydon/KINGSTON34/phdCandidacy/thesis/imgs'
out_plot_path = r'/media/veracrypt6/projects/mer_analysis/mer/deriv/feature_plots'

xls = pd.ExcelFile('/media/veracrypt6/projects/stealthMRI/resources/excelFiles/patient_info.xlsx')
surgical_data=xls.parse('surgical', header=3)
bids_dir=r'/media/veracrypt6/projects/mer_analysis/mer/bids'
subjects=[os.path.basename(x) for x in glob.glob(os.path.join(os.path.dirname(bids_dir),'deriv','features2','*')) if os.path.isdir(os.path.join(os.path.dirname(bids_dir),'deriv','features',x))]

subplot_letter = [chr(i) for i in range(ord('a'),ord('h')+1)]


for isubject in subjects:
	isub=int(''.join([x for x in isubject if x.isnumeric()]))
	sub_data=surgical_data[surgical_data['subjectNumber']==isub]
	if sub_data['target'].values[0].lower() =='stn':
		feature_files=glob.glob(os.path.join(os.path.dirname(bids_dir),'deriv','features2',isubject,'*'))
		for ifeat in feature_files:
			with open(ifeat, "rb") as file:
				feats = pickle.load(file)
			
			class_labels=np.zeros((1, feats.shape[0]))[0]
			for iside in np.unique(feats['side']):
				side_data=feats[feats['side']==iside]
				if any(x in iside for x in ('rt','right')):
					side_label='right'
				else:
					side_label = 'left'
				
				for ichan in np.unique(side_data['chan']):
					fileName=f"{isubject}_side-{side_label.lower()}_channel-{channelLabels[ichan].lower()}_features"
					
					dorsal=sub_data[f'Surg{side_label.title()[0]}{ichan.title()}In'].values[0]
					ventral=sub_data[f'Surg{side_label.title()[0]}{ichan.title()}Out'].values[0]
					min_idx=feats[(feats['side']==iside) & (feats['chan']==ichan)]['depth'].index[0]
					depths=feats[(feats['side']==iside) & (feats['chan']==ichan)]['depth'].to_numpy().astype(float)
					
					ax_cnt = 7
					firstPlot=True
					plot_cnt=1
					
					for ifeature in range(feats.iloc[:,4:].shape[1]):
						if ax_cnt == 7:
							if not firstPlot:
								fig.suptitle(f"{isubject} {side_label.lower()} side: {channelLabels[ichan].lower()} channel", y = 0.98, fontsize=22, fontweight='bold')
								plt.tight_layout(pad=2)
								
								if not os.path.exists(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}.svg")):
									plt.savefig(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}.svg"),transparent=True,dpi=400)
									plt.savefig(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}.png"),transparent=True,dpi=400)
									plt.savefig(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}_white.png"),transparent=False,dpi=400)
								plt.close()
								plot_cnt +=1
								
							nrow = 4; ncol = 2;
							plt.ioff()
							fig, axs = plt.subplots(nrows=nrow, ncols=ncol,figsize=(20,16))
							axs=axs.reshape(-1)
							ax_cnt = 0
							firstPlot=True
						else:
							firstPlot=False
							ax_cnt += 1
						
						feature = feats[(feats['chan']==ichan)].iloc[:,ifeature+4].to_numpy().astype(float)
						
						sns.lineplot(x=np.arange(0, len(depths), 1),y=feature, color='black', linewidth=2, ax=axs[ax_cnt])
						axs[ax_cnt].xaxis.set_ticks(np.arange(0, len(depths), 1))
						axs[ax_cnt].xaxis.set_ticklabels(['{:.2f}'.format(x) for x in depths],rotation=45, ha="right",rotation_mode="anchor")
						axs[ax_cnt].set_ylabel(feature_label_dict[list(feats)[ifeature+4]].title(), fontsize=14, fontweight='bold')
						axs[ax_cnt].tick_params(axis='both', which='major', labelsize=12)
						axs[ax_cnt].yaxis.set_label_coords(-0.08,.5)
						axs[ax_cnt].set_xlim(0,len(depths)-1)
						
						tickLocs_x=axs[ax_cnt].get_yticks()
						cadenceX= tickLocs_x[2] - tickLocs_x[1]
						axs[ax_cnt].set_ylim(tickLocs_x[0],tickLocs_x[-1])
						tickLabels=['{:.2f}'.format(x) for x in tickLocs_x]
						axs[ax_cnt].set_yticks(tickLocs_x, minor=False), axs[ax_cnt].set_yticklabels(tickLabels)
						yfmt = ScalarFormatterForceFormat()
						yfmt.set_powerlimits((-2,2))
						axs[ax_cnt].yaxis.set_major_formatter(yfmt)
						axs[ax_cnt].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
						axs[ax_cnt].grid()
						
						axs[ax_cnt].text(-.15, 1,f'{subplot_letter[ax_cnt]})', transform=axs[ax_cnt].transAxes, fontsize=15, fontweight='bold')
						
						if not np.isnan(dorsal) and not np.isnan(ventral):
							idx_dor,val_dor = min(enumerate(depths), key=lambda x: abs(x[1]-dorsal))
							idx_ven,val_ven = min(enumerate(depths), key=lambda x: abs(x[1]-ventral))
							
							axs[ax_cnt].axvline(idx_dor, color='#4daf4a', linewidth=2,zorder=11,label='Dorsal Border')
							axs[ax_cnt].axvline(idx_ven, color='#e41a1c', linewidth=2,zorder=11,label='Ventral Border')
							
							axs[ax_cnt].add_patch(mpatches.Rectangle((idx_dor,tickLocs_x[0]), idx_ven-idx_dor, tickLocs_x[-1]+abs(tickLocs_x[0]), alpha=.4, facecolor='#dede00', zorder=10))
					
					fig.suptitle(f"{isubject} {side_label.lower()} side: {channelLabels[ichan].lower()} channel", y = 0.98, fontsize=22, fontweight='bold')
					plt.tight_layout(pad=2)
					
					while ax_cnt <7:
						ax_cnt += 1
						fig.delaxes(axs[ax_cnt])
					
					if not os.path.exists(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}.svg")):
						plt.savefig(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}.svg"),transparent=True,dpi=400)
						plt.savefig(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}.png"),transparent=True,dpi=400)
						plt.savefig(os.path.join(out_plot_path, fileName + f"_0{plot_cnt}_white.png"),transparent=False,dpi=400)
					plt.close()
		
		print(f'Finished {isubject}')

#%%
downsampleFactor = 5

in_dir=r'/home/greydon/Downloads/mer_figure'
nucleus=['thal','zi','STN','snr']
sigbufs=[]
for inuc in nucleus:
	iedf=glob.glob(in_dir+f'/*{inuc}.edf')[0]
	
	f = pyedflib.EdfReader(iedf)
	annots=f.readAnnotations()
	num_chans = f.signals_in_file
	n_samps = f.getNSamples()[0]
	sf = f.getSampleFrequencies()/downsampleFactor
	side_label=f.admincode.decode('latin-1')
	for i in np.arange(num_chans):
		tempdat = mer.downsample(f.readSignal(i), sf[i]*downsampleFactor,sf[i])
		tempdat = mer.butterBandpass(tempdat, lowcut = 700, highcut =3000, fs = sf[i], order = 3)
		sigbufs.append(tempdat)
	
	f.close()


min_len = len(min(sigbufs, key=len))
max_len = len(max(sigbufs, key=len))
feature=np.vstack([x[200:min_len] for x in sigbufs])

feature[1,:] *= 10
feature[3,:] *= 2


fig, ax = plt.subplots(figsize=(14,8))
ax.plot(feature.T + 5 * np.arange(0, 4, 1), color='black', linewidth=0.2)
ax.yaxis.set_ticks(5 * np.arange(0, 4, 1))
ax.yaxis.set_ticklabels([])
ax.xaxis.set_ticks([])
start, end = ax.get_xlim()
xTickLabs = np.arange(0, len(feature.T)+1, int(len(feature.T)/5))/sf[0]
ax.xaxis.set_ticklabels([])
ax.set_ylim(-5,(4*5))
ax.set_xlim(0,len(feature.T))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.gca().invert_yaxis()

ax.text(-.3,.79,'Thalamus', transform=ax.transAxes, fontsize=18, fontweight='bold')
ax.text(-.3,.59,'Zona incerta', transform=ax.transAxes, fontsize=18, fontweight='bold')
ax.text(-.3,.39,'Subthalamic nucleus', transform=ax.transAxes, fontsize=18, fontweight='bold')
ax.text(-.3,.19,'Substantia nigra\npars compacta', transform=ax.transAxes, fontsize=18, fontweight='bold')

ax.annotate('', xy=(0,.15), xycoords='axes fraction', xytext=(.2,.15), arrowprops=dict(arrowstyle="-", color='black',linewidth=2),transform=ax.transAxes)
ax.text(.06,.12,'0.5 sec', transform=ax.transAxes, fontsize=14, fontweight='bold')


plt.tight_layout()
plt.subplots_adjust(left=0.25)

#%%

fileName="mer_signals"
plt.savefig(os.path.join(out_path,f"{fileName}.svg"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"{fileName}.png"),transparent=True,dpi=500)
plt.savefig(os.path.join(out_path,f"{fileName}_white.png"),transparent=False,dpi=500)
plt.close()

#%%

out_path='/media/greydon/KINGSTON34/phdCandidacy/thesis/imgs'

xls = pd.ExcelFile('/media/veracrypt6/projects/stealthMRI/resources/excelFiles/patient_info.xlsx')
surgical_data=xls.parse('surgical', header=3)
bids_dir=r'/media/veracrypt6/projects/mer_analysis/mer/bids'
subjects=[os.path.basename(x) for x in glob.glob(os.path.join(os.path.dirname(bids_dir),'deriv','features2','*')) if os.path.isdir(os.path.join(os.path.dirname(bids_dir),'deriv','features',x))]

final_data=[]
for isubject in subjects:
	isub=int(''.join([x for x in isubject if x.isnumeric()]))
	sub_data=surgical_data[surgical_data['subjectNumber']==isub]
	if sub_data['target'].values[0].lower() =='stn':
		feature_files=glob.glob(os.path.join(os.path.dirname(bids_dir),'deriv','features',isubject,'*'))
		for ifeat in feature_files:
			with open(ifeat, "rb") as file:
				feats = pickle.load(file)
			
			class_labels=np.zeros((1, feats.shape[0]))[0]
			for iside in np.unique(feats['side']):
				side_data=feats[feats['side']==iside]
				for ichan in np.unique(side_data['chan']):
					dorsal=sub_data[f'Surg{iside.title()[0]}{ichan.title()}In'].values[0]
					ventral=sub_data[f'Surg{iside.title()[0]}{ichan.title()}Out'].values[0]
					min_idx=feats[(feats['side']==iside) & (feats['chan']==ichan)]['depth'].index[0]
					depths=feats[(feats['side']==iside) & (feats['chan']==ichan)]['depth'].to_numpy().astype(float)
					
					if not np.isnan(dorsal) and not np.isnan(ventral):
						idx_dor,val_dor = min(enumerate(depths), key=lambda x: abs(x[1]-dorsal))
						idx_ven,val_ven = min(enumerate(depths), key=lambda x: abs(x[1]-ventral))
						idx_dor+=min_idx
						idx_ven+=min_idx
						
						class_labels[idx_dor:idx_ven+1]=1
			
			feats.insert(4, 'Class', class_labels.astype(int))
			final_data.append(feats)


# remove uneccesary columns
final_dataframe = pd.concat(final_data).reset_index(drop=True)
final_dataframe['subject']=final_dataframe['subject'].str.extract('(\d+)', expand=False)
final_dataframe= final_dataframe.drop(['side','chan','depth','mavSlope'], axis=1)
final_dataframe = clean_dataset(final_dataframe).dropna()
final_dataframe['subject']=final_dataframe['subject'].astype(int)

# put all the STN observations in a separate dataset
within_STN = final_dataframe.loc[final_dataframe['Class'] == 1]

# randomly select equal num of observations from npn-STN class (majority class)
outside_STN = final_dataframe.loc[final_dataframe['Class'] == 0].sample(n=within_STN.shape[0],random_state=42)

# concatenate both dataframes again
final_dataframe = pd.concat([within_STN, outside_STN]).reset_index(drop=True)
final_dataframe = final_dataframe.sort_values(by=['subject'], ascending=True)

# remove redundant feature columns
final_dataframe=remove_correlated_features(final_dataframe, .90)
final_dataframe=backward_elimination(final_dataframe, .15)

#final_dataframe = outlier_removal_IQR(final_dataframe)


groups_holdout = np.random.choice(final_dataframe['subject'].unique(), 3)

# seperate the labels from the features
X = final_dataframe[~final_dataframe['subject'].isin(groups_holdout)].drop(['Class','subject'], axis=1).reset_index(drop=True)
y = final_dataframe[~final_dataframe['subject'].isin(groups_holdout)]['Class'].astype(int).reset_index(drop=True)
groups = final_dataframe[~final_dataframe['subject'].isin(groups_holdout)]['subject'].astype(int).reset_index(drop=True).to_numpy()

X_holdout = final_dataframe[final_dataframe['subject'].isin(groups_holdout)].drop(['Class','subject'], axis=1).reset_index(drop=True)
y_holdout = final_dataframe[final_dataframe['subject'].isin(groups_holdout)]['Class'].astype(int).reset_index(drop=True)

Counter(y)


# prep for plotting
data_plot = (X - X.mean()) / (X.std())
data_plot = pd.concat([y, data_plot],axis=1)
data_melt = pd.melt(data_plot, id_vars="Class",var_name="features",value_name='value')



#%% Feature Importance


# split data train 70 % and test 30 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


steps = [('scaler', MinMaxScaler()), ('RFC', RandomForestClassifier(n_estimators=1000, oob_score = True, bootstrap = True))]
pipeline = Pipeline(steps)

importances = drop_col_feat_imp(pipeline, X_train, y_train, X_test, y_test)
importances.sort_values(by=['Importance'], ascending=False,inplace=True)


#%% Correlation matrix and feature importance plot


fig = plt.figure(figsize=(12,14))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

colors=[(68,119,170),(102,204,238),(34,136,51),(204,187,68),(238,102,119),(170,51,119),(204,51,17),(222,143,5),(213,94,0)]
colors=[list(np.array(x)/255) for x in colors]

# Plot the feature importances of the forest
sns.barplot(x=importances['Importance'], y=importances.index, ax=ax1, palette=("colorblind"))
ax1.set_title("Drop-column feature importance", fontweight='bold',fontsize=22,y=1.03)
ax1.set_xlabel('Feature Importance', fontweight='bold',fontsize=18)
ax1.set_ylabel('Feature Names', fontweight='bold',fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
old_labels=[x.get_text() for x in ax1.get_yticklabels()]
new_labels=[feature_label_dict[x] for x in old_labels]
ax1.set_yticklabels(new_labels)

ax1.text(-.12, 1.05,'a)', fontsize=22, fontweight='bold', transform=ax1.transAxes)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


pal=sns.diverging_palette(h_neg=220, h_pos=10, s=50, l=50,sep=1,n=2,center='light', as_cmap=True)
# mask
mask = np.triu(np.ones_like(X.corr(), dtype=bool))
mask[np.triu_indices_from(mask)] = True
mask[np.diag_indices_from(mask)] = False
masked_corr = X[old_labels].corr().loc[~np.all(mask,axis=0), ~np.all(mask,axis=1)]


# plot correlation matrix
sns.heatmap(masked_corr, annot = True, fmt='.1f', linewidth=.5, linecolor='w', mask = mask,ax=ax2, vmin=-1,vmax=1,center= 0, cmap=pal, square=True, xticklabels=new_labels, yticklabels=new_labels,
			annot_kws={'fontsize':10},cbar_kws={"shrink": .7})
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_title("Correlation between microelectrode\n recording data features", fontweight='bold', fontsize=24, y=1.04)
ax2.tick_params(axis='both', which='major', labelsize=15)
locs, labels = plt.xticks()
plt.setp(labels, rotation=45,horizontalalignment='right')

ax2.text(-.38, 1,'b)', fontsize=22, fontweight='bold', transform=ax2.transAxes)
plt.tight_layout(pad=3)
plt.subplots_adjust(top=.95, right=.8)



#%%

fileName="mer_feature_importance_correlations"
plt.savefig(os.path.join(out_path,f"{fileName}.svg"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"{fileName}.png"),transparent=True,dpi=500)
plt.savefig(os.path.join(out_path,f"{fileName}_white.png"),transparent=False,dpi=500)
plt.close()


#%%

feat_labels=[feature_label_dict[x] for x in X.columns]

g=sns.pairplot(data_plot.sample(frac=.05), hue = 'Class', diag_kind='kde', palette='colorblind')

for i in range(data_plot.shape[1]-1):
	for j in range(data_plot.shape[1]-1):
		xlabel = g.axes[i][j].get_xlabel()
		ylabel = g.axes[i][j].get_ylabel()
		if xlabel in feature_label_dict.keys():
			g.axes[i][j].set_xlabel(feature_label_dict[xlabel])
		if ylabel in feature_label_dict.keys():
			g.axes[i][j].set_ylabel(feature_label_dict[ylabel])

#%%

#--- RBF Kernel
steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='rbf',C=1, random_state=0))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


steps = [('scaler', StandardScaler()), ('RFC', RandomForestClassifier(n_estimators=1000))]
pipeline = Pipeline(steps)
clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(X_train,y_train)
y_pred=clr_rf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


#%%

fileName="mer_feature_importance"
plt.savefig(os.path.join(out_path,f"{fileName}.svg"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"{fileName}.png"),transparent=True,dpi=500)
plt.savefig(os.path.join(out_path,f"{fileName}_white.png"),transparent=False,dpi=500)
plt.close()


#%%


# single SVM
steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='rbf',C=1, random_state=0))]
pipeline = Pipeline(steps)
model = pipeline.fit(X_train, y_train)
y_pred=model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
plot_confusion_matrix(model,X_test,y_test)

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))


# k-fold SVM
steps = [('Scaler',MinMaxScaler()),('SVM', SVC(kernel='rbf',C=10, random_state=0))]
pipeline = Pipeline(steps)

strtfdKFold = model_selection.StratifiedKFold(n_splits=10,shuffle=True)
kfold = strtfdKFold.split(X, y)

models=[]
scores = []
for k, (train, test) in enumerate(kfold):
	pipeline.fit(X.iloc[train, :], y.iloc[train])
	score = pipeline.score(X.iloc[test, :], y.iloc[test])
	scores.append(score)
	models.append(pipeline)
	print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y.iloc[train]), score))

print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))



scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

clf_RF = RandomForestClassifier(criterion='entropy',max_features='sqrt',random_state=42)
clf_SVCL = SVC(kernel='linear',C=1, probability=True,random_state=0)
clf_SVCR = SVC(kernel='rbf',C=1, probability=True,random_state=0)

score_list_RF = []
score_list_SVCL = []
score_list_SVCR = []

F1_list_RF = []
F1_list_SVCL = []
F1_list_SVCR = []
roc_auc_train_RF = []
roc_auc_test_RF = []
roc_auc_train_SVCL = []
roc_auc_test_SVCL = []
roc_auc_train_SVCR = []
roc_auc_test_SVCR = []
acc_test_RF=[]
acc_test_SVCL=[]
acc_test_SVCR=[]
pr_auc_train_RF=[]
pr_auc_test_RF=[]
pr_auc_train_SVCL=[]
pr_auc_test_SVCL=[]
pr_auc_train_SVCR=[]
pr_auc_test_SVCR=[]
accuracy_RF=[]
error_RF=[]
sensitivity_RF=[]
specificity_RF=[]
fpr_RF=[]
precision_RF=[]

accuracy_SVCL=[]
error_SVCL=[]
sensitivity_SVCL=[]
specificity_SVCL=[]
fpr_SVCL=[]
precision_SVCL=[]

accuracy_SVCR=[]
error_SVCR=[]
sensitivity_SVCR=[]
specificity_SVCR=[]
fpr_SVCR=[]
precision_SVCR=[]

cnt=1
#cv = LeaveOneOut()
lol=LeaveOneGroupOut()
y_true, y_pred = list(), list()
for train_ix, test_ix in lol.split(X_scaled,y,groups):
	# split data
	X_train, X_test = X_scaled[train_ix, :], X_scaled[test_ix, :]
	y_train, y_test = np.array(y)[train_ix], np.array(y)[test_ix]
	
	# fit model
	clf_RF.fit(X_train,y_train)
	clf_SVCL.fit(X_train,y_train)
	clf_SVCR.fit(X_train,y_train)
	
	#Test accuracy and F1
	score_RF = clf_RF.score(X_test,y_test)
	score_SVCL = clf_SVCL.score(X_test,y_test)
	score_SVCR = clf_SVCR.score(X_test,y_test)
	
	score_list_RF.append(score_RF)
	score_list_SVCL.append(score_SVCL)
	score_list_SVCR.append(score_SVCR)
	
	y_pred_RF = clf_RF.predict(X_test)
	y_prob_train_RF = clf_RF.predict_proba(X_train)[:, 1]
	precision_train_RF, recall_train_RF, t_train_RF = precision_recall_curve(y_train, y_prob_train_RF)
	y_prob_test_RF = clf_RF.predict_proba(X_test)[:, 1]
	precision_test_RF, recall_test_RF, t_test_RF = precision_recall_curve(y_test, y_prob_test_RF)
	
	y_pred_SVCL = clf_SVCL.predict(X_test)
	y_prob_train_SVCL = clf_SVCL.predict_proba(X_train)[:, 1]
	precision_train_SVCL, recall_train_SVCL, t_train_SVCL = precision_recall_curve(y_train, y_prob_train_SVCL)
	y_prob_test_SVCL = clf_SVCL.predict_proba(X_test)[:, 1]
	precision_test_SVCL, recall_test_SVCL, t_test_SVCL = precision_recall_curve(y_test, y_prob_test_SVCL)
	
	y_pred_SVCR = clf_SVCR.predict(X_test)
	y_prob_train_SVCR = clf_SVCR.predict_proba(X_train)[:, 1]
	precision_train_SVCR, recall_train_SVCR, t_train_SVCR = precision_recall_curve(y_train, y_prob_train_SVCR)
	y_prob_test_SVCR = clf_SVCR.predict_proba(X_test)[:, 1]
	precision_test_SVCR, recall_test_SVCR, t_test_SVCR = precision_recall_curve(y_test, y_prob_test_SVCR)
	
	F1_list_RF.append(f1_score(y_test,y_pred_RF,average='weighted'))
	F1_list_SVCL.append(f1_score(y_test,y_pred_SVCL,average='weighted'))
	F1_list_SVCR.append(f1_score(y_test,y_pred_SVCR,average='weighted'))
	
	acc_test_RF.append(accuracy_score(y_test,y_pred_RF))
	acc_test_SVCL.append(accuracy_score(y_test,y_pred_SVCL))
	acc_test_SVCR.append(accuracy_score(y_test,y_pred_SVCR))
	
	roc_auc_train_RF.append(roc_auc_score(y_train, y_prob_train_RF))
	roc_auc_test_RF.append(roc_auc_score(y_test, y_prob_test_RF))
	roc_auc_train_SVCL.append(roc_auc_score(y_train, y_prob_train_SVCL))
	roc_auc_test_SVCL.append(roc_auc_score(y_test, y_prob_test_SVCL))
	roc_auc_train_SVCR.append(roc_auc_score(y_train, y_prob_train_SVCR))
	roc_auc_test_SVCR.append(roc_auc_score(y_test, y_prob_test_SVCR))
	
	pr_auc_train_RF.append(auc(recall_train_RF, precision_train_RF))
	pr_auc_test_RF.append(auc(recall_test_RF, precision_test_RF))
	pr_auc_train_SVCL.append(auc(recall_train_SVCL, precision_train_SVCL))
	pr_auc_test_SVCL.append(auc(recall_test_SVCL, precision_test_SVCL))
	pr_auc_train_SVCR.append(auc(recall_train_SVCR, precision_train_SVCR))
	pr_auc_test_SVCR.append(auc(recall_test_SVCR, precision_test_SVCR))
	
	confusion = metrics.confusion_matrix(y_test, y_pred_SVCL)
	TP_SVCL = confusion[1, 1]
	TN_SVCL = confusion[0, 0]
	FP_SVCL = confusion[0, 1]
	FN_SVCL = confusion[1, 0]
	accuracy_SVCL.append((TP_SVCL + TN_SVCL) / float(TP_SVCL + TN_SVCL + FP_SVCL + FN_SVCL))
	error_SVCL.append((FP_SVCL + FN_SVCL) / float(TP_SVCL + TN_SVCL + FP_SVCL + FN_SVCL))
	sensitivity_SVCL.append(TP_SVCL / float(FN_SVCL + TP_SVCL))
	specificity_SVCL.append(TN_SVCL / (TN_SVCL + FP_SVCL))
	fpr_SVCL.append(FP_SVCL / float(TN_SVCL + FP_SVCL))
	precision_SVCL.append(TP_SVCL / float(TP_SVCL + FP_SVCL))

	confusion = metrics.confusion_matrix(y_test, y_pred_SVCR)
	TP_SVCR = confusion[1, 1]
	TN_SVCR = confusion[0, 0]
	FP_SVCR = confusion[0, 1]
	FN_SVCR = confusion[1, 0]
	accuracy_SVCR.append((TP_SVCR + TN_SVCR) / float(TP_SVCR + TN_SVCR + FP_SVCR + FN_SVCR))
	error_SVCR.append((FP_SVCR + FN_SVCR) / float(TP_SVCR + TN_SVCR + FP_SVCR + FN_SVCR))
	sensitivity_SVCR.append(TP_SVCR / float(FN_SVCR + TP_SVCR))
	specificity_SVCR.append(TN_SVCR / (TN_SVCR + FP_SVCR))
	fpr_SVCR.append(FP_SVCR / float(TN_SVCR + FP_SVCR))
	precision_SVCR.append(TP_SVCR / float(TP_SVCR + FP_SVCR))
	
	confusion = metrics.confusion_matrix(y_test, y_pred_RF)
	TP_RF = confusion[1, 1]
	TN_RF = confusion[0, 0]
	FP_RF = confusion[0, 1]
	FN_RF = confusion[1, 0]
	accuracy_RF.append((TP_RF + TN_RF) / float(TP_RF + TN_RF + FP_RF + FN_RF))
	error_RF.append((FP_RF + FN_RF) / float(TP_RF + TN_RF + FP_RF + FN_RF))
	sensitivity_RF.append(TP_RF / float(FN_RF + TP_RF))
	specificity_RF.append(TN_RF / (TN_RF + FP_RF))
	fpr_RF.append(FP_RF / float(TN_RF + FP_RF))
	precision_RF.append(TP_RF / float(TP_RF + FP_RF))
	
	print(f'Fold: {cnt} of {len(np.unique(groups))}')
	cnt+=1


# calculate accuracy
print(f'RF: {round(np.mean(score_list_RF),4)} / {round(np.mean(F1_list_RF),4)}')
print(f'SVCL: {round(np.mean(score_list_SVCL),4)} / {round(np.mean(F1_list_SVCL),4)}')
print(f'SVCR: {round(np.mean(score_list_SVCR),4)} / {round(np.mean(F1_list_SVCR),4)}')

#%%


accuracy_RF_str=f"{np.mean(accuracy_RF):.3f}"
error_RF_str=f"{np.mean(error_RF):.3f}"
sensitivity_RF_str=f"{np.mean(sensitivity_RF):.3f}"
specificity_RF_str=f"{np.mean(specificity_RF):.3f}"
fpr_RF_str=f"{np.mean(fpr_RF):.3f}"
precision_RF_str=f"{np.nanmean(precision_RF):.3f}"
f1_RF_str=f"{np.mean(F1_list_RF):.3f}"

accuracy_SVCL_str=f"{np.mean(accuracy_SVCL):.3f}"
error_SVCL_str=f"{np.mean(error_SVCL):.3f}"
sensitivity_SVCL_str=f"{np.mean(sensitivity_SVCL):.3f}"
specificity_SVCL_str=f"{np.mean(specificity_SVCL):.3f}"
fpr_SVCL_str=f"{np.mean(fpr_SVCL):.3f}"
precision_SVCL_str=f"{np.nanmean(precision_SVCL):.3f}"
f1_SVCL_str=f"{np.mean(F1_list_SVCL):.3f}"

accuracy_SVCR_str=f"{np.mean(accuracy_SVCR):.3f}"
error_SVCR_str=f"{np.mean(error_SVCR):.3f}"
sensitivity_SVCR_str=f"{np.mean(sensitivity_SVCR):.3f}"
specificity_SVCR_str=f"{np.mean(specificity_SVCR):.3f}"
fpr_SVCR_str=f"{np.mean(fpr_SVCR):.3f}"
precision_SVCR_str=f"{np.nanmean(precision_SVCR):.3f}"
f1_SVCR_str=f"{np.mean(F1_list_SVCR):.3f}"

values_matrix=[
	['random forest',accuracy_RF_str, error_RF_str, sensitivity_RF_str, specificity_RF_str, fpr_RF_str,precision_RF_str,f1_RF_str],
	['support vector machine linear',accuracy_SVCL_str, error_SVCL_str, sensitivity_SVCL_str, specificity_SVCL_str, fpr_SVCL_str,precision_SVCL_str,f1_SVCL_str],
	['support vector machine RBF',accuracy_SVCR_str, error_SVCR_str, sensitivity_SVCR_str, specificity_SVCR_str, fpr_SVCR_str,precision_SVCR_str,f1_SVCR_str]
]

writer = MarkdownTableWriter(
	table_name="example_table",
	headers=["model","accuracy",'error','sensitivity','specificity','false positive','precision','F1 score'],
	value_matrix=values_matrix
	)

writer.write_table()


#%%

train_fpr, train_tpr, tr_thresholds = roc_curve(all_y_train_RF, all_probs_train_RF)
test_fpr, test_tpr, te_thresholds = roc_curve(all_y_test_RF, all_probs_test_RF)


fig= plt.figure(figsize=(14,10))
ax = fig.add_subplot(121)
ax.grid()

ax.plot(train_fpr, train_tpr, label=f"AUC TRAIN = {auc(train_fpr, train_tpr):.4f}",linewidth=2)
ax.plot(test_fpr, test_tpr, label=f"AUC TEST = {auc(test_fpr, test_tpr):.4f}",linewidth=2)
ax.plot([0,1],[0,1],'k--')
ax.legend(fontsize = 14)
ax.set_xlabel("True Positive Rate", fontweight='bold',fontsize=18)
ax.set_ylabel("False Positive Rate", fontweight='bold',fontsize=18)
ax.set_title("ROC curve for test and train using random forest", fontweight='bold',fontsize=20,y=1.02)
ax.grid(color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(0, 1),ax.set_xlim(0, 1)
ax.tick_params(axis='both', which='major', labelsize=14)

train_fpr, train_tpr, tr_thresholds = roc_curve(all_y_train_SVC, all_probs_train_SVC)
test_fpr, test_tpr, te_thresholds = roc_curve(all_y_test_SVC, all_probs_test_SVC)

ax = fig.add_subplot(1, 2, 1)
ax.grid()

ax.plot(train_fpr, train_tpr, label=f"AUC TRAIN = {auc(train_fpr, train_tpr):.4f}",linewidth=2)
ax.plot(test_fpr, test_tpr, label=f"AUC TEST = {auc(test_fpr, test_tpr):.4f}",linewidth=2)
ax.plot([0,1],[0,1],'k--')
ax.legend(fontsize = 14)
ax.set_xlabel("True Positive Rate", fontweight='bold',fontsize=18)
ax.set_ylabel("False Positive Rate", fontweight='bold',fontsize=18)
ax.set_title("ROC curve for test and train using random forest", fontweight='bold',fontsize=20,y=1.02)
ax.grid(color='black', linestyle='-', linewidth=0.5)
ax.set_ylim(0, 1),ax.set_xlim(0, 1)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()


#%%

fileName="roc_curve_model"
plt.savefig(os.path.join(out_path,f"{fileName}.svg"),transparent=True,dpi=400)
plt.savefig(os.path.join(out_path,f"{fileName}.png"),transparent=True,dpi=500)
plt.savefig(os.path.join(out_path,f"{fileName}_white.png"),transparent=False,dpi=500)
plt.close()

#%%

results = np.c_[np.r_[np.repeat(1,len(score_list_RF)),np.repeat(2,len(score_list_SVC))], 
				np.r_[score_list_RF,score_list_SVC],
				np.r_[F1_list_RF,F1_list_SVC]
				]
results = pd.DataFrame(results, columns=['model','score','f1_score'])

results

fig = plt.figure()
ax = fig.add_subplot(221)
sns.boxplot(data=results,x='model', ax=ax)
ax.set_xticklabels(['Random forest','Support vector machine'])
ax.set_title("Accuracy score", fontweight='bold',fontsize=24,y=1.02)

results = np.c_[np.r_[np.repeat(1,len(F1_list_RF)),np.repeat(2,len(F1_list_SVC))], np.r_[F1_list_RF,F1_list_SVC]]
results = pd.DataFrame(results, columns=['model','score'])

ax = fig.add_subplot(222)
sns.boxplot(data=results, y='score',x='model', ax=ax)
ax.set_xticklabels(['Random forest','Support vector machine'])
ax.set_title("F1 score", fontweight='bold',fontsize=24,y=1.02)


results = np.c_[np.r_[np.repeat(1,len(roc_auc_test_RF)),np.repeat(2,len(roc_auc_test_SVC))], np.r_[roc_auc_test_RF,roc_auc_test_SVC]]
results = pd.DataFrame(results, columns=['model','score'])

ax = fig.add_subplot(223)
sns.boxplot(data=results, y='score',x='model', ax=ax)
ax.set_xticklabels(['Random forest','Support vector machine'])
ax.set_title("ROC score", fontweight='bold',fontsize=24,y=1.02)


results = list([pr_auc_test_RF,pr_auc_test_SVC])

ax = fig.add_subplot(224)
sns.boxplot(results, ax=ax)
ax.set_xticklabels(['Random forest','Support vector machine'])
ax.set_title("Precision score", fontweight='bold',fontsize=24,y=1.02)








