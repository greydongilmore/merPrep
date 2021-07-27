#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 21:20:52 2021

@author: greydon
"""
from bids import BIDSLayout
import pyedflib
import numpy as np
import re
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import os
from multiprocessing import Pool
from functools import partial
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import glob
import pickle
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import merPrep as mer

def compute_features(chan_labels, sigbufs, annots, ichan):
	index=[i for i,x in enumerate(chan_labels) if x == ichan][0]
	depths=[float(re.findall(r"[-+]?\d*\.\d+|\d+",x)[0]) for x in annots[2]]
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
			mer.meanFrq(tempData,24000),
			mer.powerAVG(mer.butterBandpass(tempData, lowcut = 500, highcut = 1000, fs = 24000, order = 5)),
			mer.powerAVG(mer.butterBandpass(tempData, lowcut = 1000, highcut = 3000, fs = 24000, order = 5)),
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

def remove_correlated_features(X):
	corr_threshold = 0.9
	corr = X.corr()
	drop_columns = np.full(corr.shape[0], False, dtype=bool)
	for i in range(corr.shape[0]):
		for j in range(i + 1, corr.shape[0]):
			if corr.iloc[i, j] >= corr_threshold:
				drop_columns[j] = True
	columns_dropped = X.columns[drop_columns]
	X.drop(columns_dropped, axis=1, inplace=True)
	print(f"Columns dropped: {columns_dropped.values}")
	return X

def backward_elimination(X,y):
	#Adding constant column of ones, mandatory for sm.OLS model
	X_1 = sm.add_constant(X)#Fitting sm.OLS model
	model = sm.OLS(y,X_1).fit()
	model.pvalues
	
	#Backward Elimination
	cols = list(X.columns)
	pmax = 1
	while (len(cols)>0):
		p= []
		X_1 = X[cols]
		X_1 = sm.add_constant(X_1)
		model = sm.OLS(y,X_1).fit()
		p = pd.Series(model.pvalues.values[1:],index = cols)
		pmax = max(p)
		feature_with_p_max = p.idxmax()
		if(pmax>0.1):
			cols.remove(feature_with_p_max)
		else:
			break
	
	selected_features_BE = cols
	X = X[selected_features_BE]
	print(f"Columns kept: {selected_features_BE}")
	return X


#%%

bids_dir=r'/media/veracrypt6/projects/mer_analysis/mer/bids'

layout = BIDSLayout(bids_dir)

cols=['subject','side','chan','depth','mav','mavSlope','variance','mmav1','mmav2','rms','curveLength','zeroCross','threshold','wamp','ssi',
				 'power','peaks','tkeoTwo','tkeoFour','shapeF','kurtosis','skew','meanF','AvgPowerMU','AvgPowerSU','entropy','waveletStd']

ignore_subs=['P061']

for isubject in layout.get_subjects()[::-1][9:]:
	if isubject not in ignore_subs:
		edf_files=layout.get(subject=isubject, extension='.edf', return_type='filename')
		for iedf in edf_files:
			outname=os.path.join(iedf.split(f'/bids/sub-{isubject}')[0],'deriv','features',f'sub-{isubject}',os.path.splitext(os.path.basename(iedf))[0].replace('ieeg','features')+'.pkl')
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
					sigbufs[i, :] = f.readSignal(i)
					
				f.close()
				
				signal_features=[]
				pool = Pool(5)
				func = partial(compute_features, chan_labels,sigbufs,annots)
				
				for result in pool.imap(func, chan_labels):
					signal_features.append(result)
				
				df1 = pd.DataFrame(np.vstack(signal_features), columns=cols)
				df1.to_pickle(outname)

#%%

xls = pd.ExcelFile('/media/veracrypt6/projects/stealthMRI/resources/excelFiles/patient_info.xlsx')
surgical_data=xls.parse('surgical', header=3)

bids_dir=r'/media/veracrypt6/projects/mer_analysis/mer/bids'
subjects=[os.path.basename(x) for x in glob.glob(os.path.join(os.path.dirname(bids_dir),'deriv','features','*')) if os.path.isdir(os.path.join(os.path.dirname(bids_dir),'deriv','features',x))]



final_data=[]
for isubject in subjects:
	isub=int(''.join([x for x in isubject if x.isnumeric()]))
	sub_data=surgical_data[surgical_data['subjectNumber']==isub]
	if sub_data['target'].values[0].lower() =='stn':
		feature_files=glob.glob(os.path.join(os.path.dirname(bids_dir),'deriv','features',isubject,'*'))
		for ifeat in feature_files:
			with open(ifeat, "rb") as file:
				feats = pickle.load(file)
			
			class_labels=np.zeros((1,feats.shape[0]))[0]
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


#%%


# remove uneccesary columns
final_dataframe = pd.concat(final_data)
final_dataframe= final_dataframe.drop(['subject','side','chan','depth','mavSlope'], axis=1).astype(float)
final_dataframe = clean_dataset(final_dataframe)

# put all the STN observations in a separate dataset
within_STN = final_dataframe.loc[final_dataframe['Class'] == 1]

# randomly select equal num of observations from npn-STN class (majority class)
outside_STN = final_dataframe.loc[final_dataframe['Class'] == 0].sample(n=within_STN.shape[0],random_state=42)

# concatenate both dataframes again
final_dataframe = pd.concat([within_STN, outside_STN])

# seperate the labels from the features
X = final_dataframe.drop(['Class'], axis=1)
y = final_dataframe['Class'].astype(int)
Counter(y)

# remove redundant feature columns
X=remove_correlated_features(X)
X=backward_elimination(X,y)


Q1=X.quantile(0.15)
Q3=X.quantile(0.85)
iqr=50*(Q3-Q1)
for ifeat in iqr.keys():
	q1_idx=X[ifeat][X[ifeat] < Q1[ifeat]-iqr[ifeat]].index
	X.drop(q1_idx,inplace=True)
	y.drop(q1_idx,inplace=True)
	q3_idx=X[ifeat][X[ifeat] > Q3[ifeat]+iqr[ifeat]].index
	X.drop(q3_idx,inplace=True)
	y.drop(q3_idx,inplace=True)


# split data train 70 % and test 30 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)



# prep for plotting
data_plot = (X - X.mean()) / (X.std())
data_plot = pd.concat([y, data_plot],axis=1)
data_melt = pd.melt(data_plot, id_vars="Class",var_name="features",value_name='value')

plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="Class", data=data_melt)
plt.xticks(rotation=90)


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data_plot.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)



# split data train 70 % and test 30 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


#--- RBF Kernel
steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='rbf',C=100,  gamma=0.7, random_state=0))]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


steps = [('scaler', StandardScaler()), ('RFC', RandomForestClassifier(n_estimators=1000,C=10))]
pipeline = Pipeline(steps)
clf_rf = RandomForestClassifier(random_state=43)
clr_rf = clf_rf.fit(X_train,y_train)
y_pred=clr_rf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

feature_imp = pd.Series(clr_rf.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
plot_confusion_matrix(clr_rf,X_test,y_test)

random_grid = {
	'n_estimators': np.linspace(100, 1000, int((1000-100)/200) + 1, dtype=int),
	'max_depth': [1, 5, 10, 20, 50, 75, 100],
	'min_samples_split': [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)],
	'min_samples_leaf': [1, 2, 3, 4],
}

rf_base = RandomForestClassifier()
rf_random = GridSearchCV(estimator = rf_base,
							   param_grid = random_grid,
							   verbose = 2)
rf_random.fit(X_train,y_train)

metrics.accuracy_score(y_test,clf_rf.predict(X_test))
confusion_matrix(y_test,clf_rf.predict(X_test))

param_grid = {'C': [.1, 1, 10, 100], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

print(grid.best_estimator_)



kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)

scoring = 'accuracy'
steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='rbf',C=1, random_state=0))]
pipeline = Pipeline(steps)

results = model_selection.cross_val_score(pipeline, X, y, cv=kfold, scoring=scoring)



steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='rbf',C=1, random_state=0))]
pipeline = Pipeline(steps)
model = pipeline.fit(X_train, y_train)
y_pred=model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
plot_confusion_matrix(model,X_test,y_test)

print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))


steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='rbf',C=10, random_state=0))]
pipeline = Pipeline(steps)

strtfdKFold = model_selection.StratifiedKFold(n_splits=10)
kfold = strtfdKFold.split(X_train, y_train)


models=[]
scores = []
for k, (train, test) in enumerate(kfold):
	pipeline.fit(X_train.iloc[train, :], y_train.iloc[train])
	score = pipeline.score(X_train.iloc[test, :], y_train.iloc[test])
	scores.append(score)
	models.append(pipeline)
	print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))

print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

