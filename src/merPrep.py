# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:47:20 2018

@author: Greydon
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import welch, hanning, butter, lfilter, resample
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
import pywt
import tables
import subprocess
import scipy.io as spio
import h5py
import json


##############################################################################
#                              HELPER FUNCTIONS                              #
##############################################################################
def sorted_nicely(data, reverse = False):
	"""
	Sorts the given iterable in the way that is expected.
	
	Parameters
	----------
		data: array-like
			The iterable to be sorted.
	
	Returns
	-------
		The sorted list.
	"""
	
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	
	return sorted(data, key = alphanum_key, reverse=reverse)

def downsample(data, oldFS, newFS):
	"""
	Resample data from oldFS to newFS using the scipy 'resample' function.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		oldFS: int
			The sampling frequency of the data.
		newFS: int
			The new sampling frequency.
	
	Returns
	-------
		newData: array-like
			The downsampled dataset.
	"""

	newNumSamples = int((len(data) / oldFS) * newFS)
	newData = pd.DataFrame(resample(data, newNumSamples))
	
	return newData

##############################################################################
#                                 FILTERS                                    #
##############################################################################
def butter_bandpass(lowcut, highcut, fs, order):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	
	return b, a
	   
def butterBandpass(d, lowcut, highcut, fs, order):
	b, a = butter_bandpass(lowcut, highcut, fs, order)
	y = lfilter(b, a, d)
	
	return y

##############################################################################
#                            TIME DOMAIN FEATURES                            #
##############################################################################
def MAV(data):
	"""
	Mean absolute value: the average of the absolute value of the signal.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		MAVData: 1D numpy array containing average absolute value
		
	Reference 
	---------
		Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy 
		for multifunction myoelectric control. IEEE Transactions on 
		Bio-Medical Engineering, 40(1), 82–94. 
	"""
	
	MAVData = sum(abs(data))/len(data)
	
	return MAVData

def MAVS(data1, data2):
	"""
	Mean Absolute Value Slope: the difference between MAVs in adjacent 
		segments.
	
	Parameters
	----------
		data1: array-like
			2D matrix of shape (time, data)
		data2: array-like
			2D matrix of shape (time, data) of subsequent segment to x1
		
	Returns
	-------
		MAVSlope: 1D numpy array containing MAV for adjacent signals
		
	Reference
	---------
		Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy 
		for multifunction myoelectric control. IEEE Transactions on 
		Bio-Medical Engineering, 40(1), 82–94.
	"""
	
	MAV1Data = sum(abs(data1))/len(data1)
	MAV2Data = sum(abs(data2))/len(data2)
	
	MAVSlope = MAV2Data - MAV1Data
	
	return MAVSlope

def MMAV1(data):
	"""
	Modified Mean Absolute Value 1: an extension of MAV using a weighting 
		window function on data below 25% and above 75%.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		MMAV1Data: 1D numpy array containing modified MAV for given signal
	
	Reference
	---------
		Phinyomark, A., Limsakul, C., & Phukpattaranont, P. (2009). 
		A Novel Feature Extraction for Robust EMG Pattern Recognition. Journal 
		of Medical Engineering and Technology, 40(4), 149–154.
	"""
	
	w1 = 0.5
	segment = int(len(data)*0.25)
	start = abs(data[0:segment,])*w1
	middle = abs(data[segment:(len(data)-segment),])
	end = abs(data[(len(data)-segment):,])*w1
	
	combined = np.concatenate((start, middle, end))
	MMAV1Data = sum(abs(combined))/len(combined)
	
	return MMAV1Data
	
def MMAV2(data):
	"""
	Modified Mean Absolute Value 2: the smooth window is improved by using 
		a continuous weighting window function on data below 25% and above 75%.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		MMAV2Data: 1D numpy array containg modified MAV for signal
	
	Reference
	---------
		Phinyomark, A., Limsakul, C., & Phukpattaranont, P. (2009). 
		A Novel Feature Extraction for Robust EMG Pattern Recognition. Journal 
		of Medical Engineering and Technology, 40(4), 149–154.
	"""
	
	segment = int(len(data)*0.25)
	a = []
	b = []
	for i in range(segment):
		endIdx = (len(data)-segment)+i
		a.append((4*i)/len(data))
		b.append((4*(len(data)-endIdx))/len(data))
		
	start = abs(data[0:segment,])*a
	middle = abs(data[segment:(len(data)-segment),])
	end = abs(data[(len(data)-segment):,])*b
	
	combined = np.concatenate((start,middle,end))
	MMAV2Data = sum(abs(combined))/len(combined)
	
	return MMAV2Data

def RMS(data):
	"""
	Root mean square: the root mean square of a given recording.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		RMSData: 1D numpy array containing root mean square of the signal
	
	Reference
	---------
		Phinyomark, A., Limsakul, C., & Phukpattaranont, P. (2009). 
		A Novel Feature Extraction for Robust EMG Pattern Recognition. Journal 
		of Medical Engineering and Technology, 40(4), 149–154.
	"""    
	
	RMSData = (sum(data*data)/len(data))**0.5
	
	return RMSData

def VAR(data):
	"""
	Variance: deviation of the signal from it's mean.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		varianceData: 1D numpy array containg the signal variance
	
	Reference
	---------
		Huang, H., & Chiang, C. (2000). DSP-based controller for a 
		multi-degree prosthetic hand. Robotics and Automation, 2000. …, 
		2(April), 1378–1383.
	"""
	
	meanData = sum(data)/len(data)
	varianceData = sum((data-meanData)*(data-meanData))/len(data)
	
	return varianceData

def curveLen(data):
	"""
	Curve length: the cumulative length of the waveform over the time segment.
		This feature is related to the waveform amplitude, frequency and time.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		curveLenData: 1D numpy array containing the average curve length for 
					given signal
	
	Reference
	---------
		Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy 
		for multifunction myoelectric control. IEEE Transactions on 
		Bio-Medical Engineering, 40(1), 82–94.
	"""
	
	data1 = data[1:]
	data2 = data[:-1]
	
	curveLenData = sum(abs(data2-data1))/(len(data)-1)
	
	return curveLenData

def zeroCross(data, threshold):
	"""
	Zero crossings: Calculates the number of times the signal amplitude 
		crosses the zero y-axis. 
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		zeroCrossData: 1D numpy array containing total number of zero crossings
					 in the given signal
			
	Reference
	---------
		Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy 
		for multifunction myoelectric control. IEEE Transactions on 
		Bio-Medical Engineering, 40(1), 82–94.
	"""
	
	i = abs(data[:-1]-data[1:]) > threshold
	ind = data[np.nonzero(i)[0]]
	
	zeroCrossData = len(np.where(np.diff(np.sign(ind)))[0])
	
	return zeroCrossData

def slopeSign(data):
	"""
	Slope Sign Change: The number of changes between positive and negative
		slope among three consecutive segments are performed
		with the threshold function. 
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		slopeSignData: 1D numpy array containing the total slope sign changes
					 for a given signal
			
	Reference
	---------
		Hudgins, B., Parker, P., & Scott, R. N. (1993). A new strategy 
		for multifunction myoelectric control. IEEE Transactions on 
		Bio-Medical Engineering, 40(1), 82–94.
	"""
	
	i = (data[1:-1]-data[:-2])
	j = (data[1:-1]-data[2:]) 
	
	slopeSignData = len(np.where((i*j) > 10)[0])
	
	return slopeSignData

def threshold(data):
	"""
	Threshold: measure of how scattered the sign is (deviation).
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		thresholdData: 1D numpy array containing the total threshold value for a 
				 given signal
	"""
	
	i = data-(sum(data)/len(data))
	j = sum(i*i)
	
	thresholdData = (3*(j**(1/2)))/(len(data)-1)
	
	return thresholdData

def WAMP(data, threshold):
	"""
	Willison Amplitude: the number of times that the difference between signal 
		amplitude among two adjacent segments that exceeds a predefined 
		threshold to reduce noise effects.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		threshold: int
			threshold level in uV (generally use 10 microvolts)
		
	Returns
	-------
		WAMPData: 1D numpy array containing total number of times derivative 
				was above threshold in a given signal
				
	Reference
	---------
		Huang, H., & Chiang, C. (2000). DSP-based controller for a 
		multi-degree prosthetic hand. Robotics and Automation, 2000. …, 
		2(April), 1378–1383.
	"""
	
	i = abs(data[:-1]-data[1:])
	j = i[i > threshold]
	
	WAMPData = len(j)
	
	return WAMPData

def SSI(data):
	"""
	Simple Square Integral: uses the energy of signal as a feature.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		SSIData: 1D numpy array containing the summed absolute square of the 
			   given signal
		
	Reference
	---------
		Phinyomark, A., Limsakul, C., & Phukpattaranont, P. (2009). 
		A Novel Feature Extraction for Robust EMG Pattern Recognition. Journal 
		of Medical Engineering and Technology, 40(4), 149–154.
	"""
	
	SSIData = sum(abs(data*data))
	
	return SSIData

def powerAVG(data):
	"""
	Average power: the amount of work done, amount energy transferred per 
		unit time.
		
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		powerAvgData: 1D numpy array containing average power in a given signal
	"""
	
	powerAvgData = sum(data*data)/len(data)

	return powerAvgData

def peaksNegPos(data):
	"""
	Peaks: the number of positive peaks in the data window per unit time.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		peaksNegPosData: 1D numpy array containing total number of peaks in given 
				 signal
	"""
	
	sign = lambda z: (1, -1)[z < 0]
		
	i = [sign(z) for z in (data[2:]-data[1:-1])]
	j = [sign(z) for z in (data[1:-1]-data[:-2])]
	k = [a_i - b_i for a_i, b_i in zip(i, j)]
	
	peaksNegPosData = [max([0,z]) for z in k]
	peaksNegPosData = sum(peaksNegPosData)/(len(data)-2)
	
	return peaksNegPosData

def peaksPos(data):
	"""
	Peak Density: calculates the density of peaks within the current locality. 
		A peak is defined as a point higher in amplitude than the two points 
		to its left and right side. 
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		peaksPosData: 1D numpy array containing the average number of peaks 
					   in a given signal
	"""

	data1 = data[1:-1]
	data2 = data[0:-2]
	data3 = data[2:]
	data4 = data1 - data2
	data5 = data1 - data3
	peakcount = 0
	
	for i in range(len(data)-2):
		if data4[i] > 0 and data5[i]>0:
			peakcount += 1
	
	peaksPosData = peakcount/(len(data)-2)
	
	return peaksPosData

def tkeoTwo(data):
	"""
	Teager-Kaiser Energy Operator: is analogous to the total 
		(kinetic and potential) energy of a signal. This variation uses 
		the second derivative.

	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		tkeoTwoData: 1D numpy array containing total teager energy of a given
				signal using two samples
			
	Reference
	---------
		1. Kaiser, J. F. (1990). On a simple algorithm to calculate the 
			“energy” of a signal. In International Conference on Acoustics, 
			Speech, and Signal Processing (Vol. 2, pp. 381–384). IEEE.
		2. Li, X., Zhou, P., & Aruin, A. S. (2007). Teager-Kaiser energy 
			operation of surface EMG improves muscle activity onset detection. 
			Annals of Biomedical Engineering, 35(9), 1532–8.
	"""

	i = data[1:-1]*data[1:-1]
	j = data[2:]*data[:-2]

	tkeoTwoData = sum(i-j)/(len(data)-2)

	return tkeoTwoData

def tkeoFour(data):
	"""
	Teager-Kaiser Energy Operator: is analogous to the total 
	   (kinetic and potential) energy of a signal. This variation uses 
	   the 4th order derivative.
	 
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		tkeoFourData: 1D numpy array containing total teager energy of a given
				signal using 4 samples

	Reference
	---------
		1. Kaiser, J. F. (1990). On a simple algorithm to calculate the 
			“energy” of a signal. In International Conference on Acoustics, 
			Speech, and Signal Processing (Vol. 2, pp. 381–384). IEEE.
		2. Deburchgraeve, W., Cherian, P. J., De Vos, M., Swarte, R. M., 
			Blok, J. H., Visser, G. H., … Van Huffel, S. (2008). Automated 
			neonatal seizure detection mimicking a human observer reading EEG. 
			Clinical Neurophysiology : Official Journal of the International 
			Federation of Clinical Neurophysiology, 119(11), 2447–54.               
	"""
	
	l = 1
	p = 2
	q = 0
	s = 3
	
	tkeoFourData = sum(data[l:-p]*data[p:-l]-data[q:-s]*data[s:])/(len(data)-3)

	return tkeoFourData
 
def KUR(data):
	"""
	Kurtosis: calculates the degree to which the signal has 'tails'. Heavy-tail
		would mean many outliers. A normal distribution kurtosis value is 3.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		kurtosisData: 1D numpy array containing the total kurtosis for a given signal
	
	Reference
	---------
		Choi, S., Cichocki, A., & Amari, S.-I. (2000). Flexible 
		Independent Component Analysis. Journal of VLSI Signal Processing 
		Systems for Signal, Image and Video Technology, 26(1), 25–38.
	"""
	
	meanX = sum(data)/len(data)
	diff = [z - meanX for z in data]
	sq_differences = [d**2 for d in diff]
	var = sum(sq_differences)/len(data)
	stdData = var**0.5
	
	i = sum((data-meanX)**4)
	j = (len(data)-1)*(stdData)**4
	
	kurtosisData = i/j
	
	return kurtosisData

def SKW(data):
	"""
	Skewness: measures symmetry in the signal, the data is symmetric if it 
		looks the same to the left and right of the center point. A skewness 
		of 0 would indicate absolutely no skew. 
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		skewnessData: 1D numpy array containing the total skewness for a given signal
	
	Reference
	---------
		Yiakopoulos, C. T., Gryllias, K. C., & Antoniadis, I. A. (2011). 
		Rolling element bearing fault detection in industrial environments 
		based on a K-means clustering approach. Expert Systems with 
		Applications, 38(3), 2888–2911. 
	"""
	meanX = sum(data)/len(data)
	diff = [z - meanX for z in data]
	sq_differences = [d**2 for d in diff]
	var = sum(sq_differences)/len(data)
	stdX = var**0.5
	
	i = sum((data-meanX)**3)
	j = (len(data)-1)*(stdX)**3
	
	skewnessData = i/j
	
	return skewnessData

def crestF(data):
	"""
	Crest factor: the relation between the peak amplitude and the RMS of the 
		signal.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		crestFactorData: 1D numpy array containing the total crest factor for a given
				  signal
	
	Reference
	---------
		Yiakopoulos, C. T., Gryllias, K. C., & Antoniadis, I. A. (2011). 
		Rolling element bearing fault detection in industrial environments 
		based on a K-means clustering approach. Expert Systems with 
		Applications, 38(3), 2888–2911.
	"""
	DC_remove = data - (sum(data)/len(data))
	peakAmp = max(abs(DC_remove))
	RMS = (sum(DC_remove*DC_remove)/len(DC_remove))**0.5
	
	crestFactorData = peakAmp/RMS
	
	return crestFactorData

def entropy(data):
	"""
	Entropy: is an indicator of disorder or unpredictability. The entropy is 
		smaller inside STN region because of its more rhythmic firing compared 
		to the mostly noisy background activity in adjacent regions.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		entropyData: 1D numpy array containing the total entropy for a given 
				   signal
				   
	Reference
	---------
		Ekštein, K., & Pavelka, T. (2004). Entropy And Entropy-based 
		Features In Signal Processing. Laboratory of Intelligent Communication 
		Systems, Dept. of Computer Science and Engineering, University of West 
		Bohemia, Plzen, Czech Republic, 1–2.
	"""
	
	ent = 0
	m = np.mean(data)
	for i in range(len(data)): 
		quo = abs(data[i] - m)
		ent = ent + (quo* np.log10(quo)) 
	
	entropyData = -ent
	
	return entropyData

def shapeFactor(data):
	"""
	Shape Factor: value affected by objects shape but is independent of its 
		dimensions.
		
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		shapeFactorData: 1D numpy array containing shape factor value for a 
					   given signal
					   
	Reference
	---------
		Yiakopoulos, C. T., Gryllias, K. C., & Antoniadis, I. A. (2011). 
		Rolling element bearing fault detection in industrial environments 
		based on a K-means clustering approach. Expert Systems with 
		Applications, 38(3), 2888–2911.
	"""
	RMS = (sum(data*data)/len(data))**0.5
	shapeFactorData = RMS/(sum(abs(data))/len(data))

	return shapeFactorData

def impulseFactor(data):
	"""
	Impulse Factor: 
		
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		impulseFactorData: 1D numpy array containing impulse factor value for a 
						 given signal
		
	Reference
	---------
		Yiakopoulos, C. T., Gryllias, K. C., & Antoniadis, I. A. (2011). 
		Rolling element bearing fault detection in industrial environments 
		based on a K-means clustering approach. Expert Systems with 
		Applications, 38(3), 2888–2911.
	"""
	impulseFactorData = max(abs(data))/(sum(abs(data))/len(data))

	return impulseFactorData

def clearanceFactor(data):
	"""
	Clearance Factor: 
		
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		
	Returns
	-------
		clearanceFactorData: 1D numpy array containing impulse factor value for a 
						 given signal
		
	Reference
	---------
		Yiakopoulos, C. T., Gryllias, K. C., & Antoniadis, I. A. (2011). 
		Rolling element bearing fault detection in industrial environments 
		based on a K-means clustering approach. Expert Systems with 
		Applications, 38(3), 2888–2911.
	"""
	clearanceFactorData = max(abs(data))/((sum(abs(data)**0.5)/len(data))**2)

	return clearanceFactorData

##############################################################################
#                            FREQUENCY DOMAIN                                #
##############################################################################
def computeFFT(data, Fs, normalize=False):
	"""
	Compute the FFT of `data` and return. Also returns the axis in Hz for 
		further plot.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		Fs: int
			Sampling frequency in Hz.
			
	Returns
	-------
		fAx: array-like
			Axis in Hz to plot the FFT.
		fftData: array-like
			Value of the fft.
	"""
	
	N = data.shape[0]
	fAx = np.arange(N/2) * Fs/N
	
	if normalize:
		Y = np.fft.fft(data)/int(len(data))
		fftData = abs(Y[range(int(len(data)/2))])
	else:
		Y = np.abs(np.fft.fft(data))
		fftData = 2.0/N * np.abs(Y[0:N//2])
		
	return fAx, fftData

def wrcoef(data, coef_type, coeffs, wavename, level):
	N = np.array(data).size
	a, ds = coeffs[0], list(reversed(coeffs[1:]))

	if coef_type =='a':
		return pywt.upcoef('a', a, wavename, level=level)[:N]
	elif coef_type == 'd':
		return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
	else:
		raise ValueError("Invalid coefficient type: {}".format(coef_type))

def wavlet(data, nLevels, waveletName, timewindow, windowSize, Fs):
	"""
	Wavelet Transform: captures both frequency and time information. 
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		nLevels: int 
			Number of levels for the wavlet convolution
		waveletName: str
			Name of the wavelet to be used
		timewindow: boolean
			Option to split the given signal into discrete time bins
		windowSize: int
			If timewindow is TRUE then provide the size of the time 
					 window
		Fs: int
			If timewindow is TRUE then provide the sampling rate of the given 
			signal
			
	Returns
	-------
		waveletData: 1D numpy array containing the standard deviation of the 
				   wavelet convolution for a given signal
	"""
	if timewindow == True:
		windowsize = windowSize*Fs
		n = int(len(data))
		windown=int(np.floor(n/windowsize))
		waveletData=[]
		for i in range(windown-1):
			xSeg = data[windowsize*i:windowsize*(i+1)]
			coeffs = pywt.wavedec(xSeg, waveletName, level=nLevels)
			waveletData.append(np.std(wrcoef(xSeg, 'd', coeffs, waveletName, nLevels)))
			
	else:
		coeffs = pywt.wavedec(data, waveletName, level=nLevels)
		waveletData = np.std(wrcoef(data, 'd', coeffs, waveletName, nLevels))

	return waveletData

def computeAvgDFFT(data, Fs, windowLength = 256, windowOverlapPrcnt = 50, Low=500, High=5000):
	"""
	Fast Fourier Transform: captures the frequency information within a signal.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		Fs: int
			Sampling rate of the given signal
		Low: int 
			The highpass frequency cutoff
		High: int 
			The lowpass frequency cutoff
		
	Returns
	-------
		averagePxxWelch: average power in defined passband
	"""
	
	# Defining hanning window
	win = hanning(windowLength, True)
	welchNoverlap = int(windowLength*windowOverlapPrcnt/100.0)
	
	f, Pxxf = welch(data, Fs, window=win, noverlap=welchNoverlap, nfft=windowLength, return_onesided=True)
	
	indexLow = np.where(f == min(f, key=lambda x:abs(x-Low)))[0][0]
	indexHigh = np.where(f == min(f, key=lambda x:abs(x-High)))[0][0]
	averagePxxWelch = np.mean(Pxxf[indexLow:indexHigh])
		
	return averagePxxWelch
	
def meanFrq(data, Fs):
	"""
	Mean Frequency: calculated as the sum of the product of the spectrogram 
		intensity (in dB) and the frequency, divided by the total sum of 
		spectrogram intensity.
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		Fs: int
			Sampling rate of the given signal
		
	Returns
	-------
		meanFrqData: 1D numpy array containing the mean frequency of a given 
				   signal
				   
	Reference
	---------
		Oskoei, M. A., & Hu, H. (2006). GA-based Feature Subset 
		Selection for Myoelectric Classification. In 2006 IEEE International 
		Conference on Robotics and Biomimetics (pp. 1465–1470). IEEE.
	"""
	win = 4 * Fs
	freqs, psd = welch(data, Fs, nperseg=win, scaling='density')
	
	meanFrqData = sum(freqs*psd)/sum(psd)
	
	return meanFrqData

def freqRatio(data, Fs):
	"""
	Frequency Ratio: ratio between power in lower frequencies and power in 
		higher frequencies
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		Fs: int
			Sampling rate of the given signal
		
	Returns
	-------
		freqRatioData:
	
	Reference
	---------
		Han, J. S., Song, W. K., Kim, J. S., Bang, W. C., Lee, H., & 
		Bien, Z. (2000). New EMG pattern recognition based on soft computing 
		techniques and its application to control of a rehabilitation robotic 
		arm. Proc. of 6th International Conference on Soft Computing 
		(IIZUKA2000), 890–897.
	"""
	win = 4 * Fs
	freqs, psd = welch(data, Fs, nperseg=win, scaling='density')    
	freqRatioData = abs(psd[:int(len(freqs)/2)])/abs(psd[int(len(freqs)/2):-1])
	
	return freqRatioData
	
def meanAmpFreq(data, windowSize, Fs):
	"""
	Mean Frequency Amplitude: 
	
	Parameters
	----------
		data: array-like
			2D matrix of shape (time, data)
		windowSize: int
			Size of the window
		Fs: int
			Sampling rate of the given signal
		
	Returns
	-------
		meanAmpFreqData: 1D numpy array containing 
	"""
	window = windowSize*Fs
	n = int(len(data))
	windown=int(np.floor(n/window))
	meanAmpFreqData=[]
	
	for i in range(windown-1):
		xSeg = data[window*i:window*(i+1)]
		meanAmpFreqData.append(np.median(abs(np.fft.fft(xSeg))))
		
	return meanAmpFreqData

##############################################################################
#                             VISUALIZATION                                  #
##############################################################################

channelLabels = {1:"Center", 2:"Anterior", 3:"Posterior", 4:"Medial", 5:"Lateral"}

class MathTextSciFormatter(mticker.Formatter):
	def __init__(self, fmt="%1.2e"):
		self.fmt = fmt
	def __call__(self, x, pos=None):
		s = self.fmt % x
		decimal_point = '.'
		positive_sign = '+'
		tup = s.split('e')
		significand = tup[0].rstrip(decimal_point)
		sign = tup[1][0].replace(positive_sign, '')
		exponent = tup[1][1:].lstrip('0')
		if exponent:
			exponent = '10^{%s%s}' % (sign, exponent)
		if significand and exponent:
			s =  r'\bf %s{\times}%s' % (significand, exponent)
		else:
			s =  r'\bf %s%s' % (significand, exponent)
		return "${}$".format(s)
	
def axFormat(a):
	a.yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
	a.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	for tick in a.xaxis.get_major_ticks():
		tick.label1.set_fontweight('bold')
#    for tick in a.yaxis.get_major_ticks():
#        tick.label1.set_fontweight('bold')
def axFormaty(a):
	a.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	a.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	for tick in a.yaxis.get_major_ticks():
		tick.label1.set_fontweight('bold')
	
def plotting(x, showOnly, timeWindow, processedFolder):
	featureLabels = pd.DataFrame([{'mav': 'Mean Absolute Value', 
								   'mavSlope': 'Mean Absolute Value Slope', 
								   'variance': 'Variance', 
								   'mmav1': 'Mean Absolute Value 1', 
								   'mmav2': 'Mean Absolute Value 2',
								   'rms': 'Root Mean Square', 
								   'curveLength': 'Curve Length', 
								   'zeroCross': 'Zero Crossings', 
								   'slopeSign': 'Slope Sign', 
								   'threshold': 'Threshold', 
								   'wamp': 'Willison Amplitude', 
								   'ssi': 'Simple Square Integral',
								   'power': 'Power', 
								   'peaksNegPos': 'Peaks - Negative and Positive', 
								   'peaksPos': 'Peaks - Positive', 
								   'tkeoTwo': 'Teager-Kaiser Energy Operator - Two Samples', 
								   'tkeoFour': 'Teager-Kaiser Energy Operator - Four Samples',
								   'kurtosis': 'Kurtosis', 
								   'skew': 'Skewness', 
								   'crestF': 'Crest Factor', 
								   'meanF': 'Mean Frequency', 
								   'binData': 'Raw Data',
								   'AvgPowerMU': 'Bandpass Power (500-1000Hz)', 
								   'AvgPowerSU': 'Bandpass Power (1000-3000Hz)', 
								   'entropy': 'Signal Entropy', 
								   'waveletStd': 'STD of Wavlet Convolution', 
								   'spikeISI': 'Inter-Spike Interval', 
								   'meanISI': 'Mean of ISI', 
								   'stdISI': 'STD of ISI', 
								   'burstIndex': 'Burst Index', 
								   'pauseIndex': 'Pause Index', 
								   'pauseRatio': 'Pause Ratio', 
								   'spikeDensity': 'Spike Density'}])
	
	subList = np.unique(x['subject'])
	
	for isub in range(len(subList)):
		
		if timeWindow==True:
			outputDir = processedFolder + '/sub-' + str(subList[isub]) + '/timeWindow/'
			if not os.path.exists(outputDir):
				os.makedirs(outputDir)
		else:
			outputDir = processedFolder + '/sub-' + str(subList[isub]) + '/depthWindow/'
			if not os.path.exists(outputDir):
				os.makedirs(outputDir)

		numSides = np.unique(x[(x['subject']==subList[isub])]['side'])
	
		for iside in range(len(numSides)):
			numChans = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['channel'])
			numFeatures = list(x.drop(['subject','side','channel','depth','labels', 'chanChosen'], axis=1))
			if np.isnan(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['chanChosen']).any():
				chanSel = np.nan
			else:
				chanSel = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['chanChosen'])
			
			for ifeatures in range(len(numFeatures)):
				if 'binData' in numFeatures[ifeatures]:
					fileName = 'sub-' + str(subList[isub]) + '_side-' + numSides[iside] + '_' + featureLabels[numFeatures[ifeatures]].values[0].replace(" ", "")
					
					plotRaw(x,subList[isub],numSides[iside], numChans, chanSel, fileName, outputDir, 24000)
					print('Finished subject', str(subList[isub]), numSides[iside], 'side', 'feature:', featureLabels[numFeatures[ifeatures]].values[0])
				elif 'spikeISI' in numFeatures[ifeatures]:
					nothing = []
				elif numFeatures[ifeatures] in {'PositiveSpikes','PositiveTimes','NegativeSpikes','NegativeTimes'}:
					nothing = []
				else:
					fig, axs = plt.subplots(len(numChans),1, sharex=True, sharey=False)
					fig.subplots_adjust(hspace=0.1, wspace=0)
					titleLab  = 'Sub-' + str(subList[isub]) + ' ' + numSides[iside] + ' Side: ' + featureLabels[numFeatures[ifeatures]].values[0]
					fileName = 'sub-' + str(subList[isub]) + '_side-' + numSides[iside] + '_' + featureLabels[numFeatures[ifeatures]].values[0].replace(" ", "")
					
					for ichan in range(len(numChans)):
						feature = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])][numFeatures[ifeatures]])
						
						depths = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['depth'])
						labels = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['labels'])
						channel = channelLabels.get(numChans[ichan])
						muA = np.mean(feature)
						
						if timeWindow==False:
							if len(numChans) ==1:
								axs.plot(depths, feature)
								axs.set_xlim(depths[0,],depths[-1])
							else:
								axs[ichan].plot(depths, feature)
								axs[ichan].set_xlim(depths[0,],depths[-1])
						else:
							if len(numChans) ==1:
								axs.plot(np.arange(0,x.shape[1],1), feature)
								axs.set_xlim(0,(feature.shape[1]))
							else:
								axs[ichan].plot(np.arange(0,x.shape[1],1), feature)
								axs[ichan].set_xlim(0,(feature.shape[1]))
						if len(numChans) ==1:
							axs.plot(axs.get_xlim(), [muA,muA], ls= 'dashed', c='black')
							if ~np.isnan(chanSel):
								if numChans[ichan] == chanSel:
									axs.annotate(channel, xy=(1.01,0.5),xycoords='axes fraction', fontsize=12, fontweight='bold', color='red')
								else:
									axs.annotate(channel, xy=(1.01,0.5),xycoords='axes fraction', fontsize=12, fontweight='bold')
							else:
								axs.annotate(channel, xy=(1.01,0.5),xycoords='axes fraction', fontsize=12, fontweight='bold')
								
							if timeWindow==False:
								xticlabs = np.arange(depths[0],depths[-1],1)
								axs.xaxis.set_ticks(xticlabs)
								axs.xaxis.set_ticklabels(xticlabs, rotation = 45)
							else:
								xticlabs = np.arange(0,len(feature),5)
								axs.xaxis.set_ticks(xticlabs)
								axs.xaxis.set_ticklabels((xticlabs*2).astype(int), rotation = 45)
							axFormat(axs)
							
							if np.size(np.where(labels==1)) != 0:
								inDepth  = depths[np.min(np.where(labels==1))]
								outDepth = depths[np.max(np.where(labels==1))]
								axs.axvspan(inDepth, outDepth, color='purple', alpha=0.2)
							for xc in depths:
								axs.axvline(x=xc, color='k', linestyle='--', alpha=0.2)
						else:
							axs[ichan].plot(axs[ichan].get_xlim(), [muA,muA], ls= 'dashed', c='black')
							
							if ~np.isnan(chanSel):
								if numChans[ichan] == chanSel:
									axs[ichan].annotate(channel, xy=(1.01,0.5),xycoords='axes fraction', fontsize=12, fontweight='bold', color='red')
								else:
									axs[ichan].annotate(channel, xy=(1.01,0.5),xycoords='axes fraction', fontsize=12, fontweight='bold')
							else:
								axs[ichan].annotate(channel, xy=(1.01,0.5),xycoords='axes fraction', fontsize=12, fontweight='bold')
							
							if timeWindow==False:
								xticlabs = np.arange(depths[0],depths[-1],1)
								axs[ichan].xaxis.set_ticks(xticlabs)
								axs[ichan].xaxis.set_ticklabels(xticlabs, rotation = 45)
							else:
								xticlabs = np.arange(0,len(feature),5)
								axs[ichan].xaxis.set_ticks(xticlabs)
								axs[ichan].xaxis.set_ticklabels((xticlabs*2).astype(int), rotation = 45)
							axFormat(axs[ichan])
						
							if np.size(np.where(labels==1)) != 0:
								inDepth  = depths[np.min(np.where(labels==1))]
								outDepth = depths[np.max(np.where(labels==1))]
								axs[ichan].axvspan(inDepth, outDepth, color='purple', alpha=0.2)
							
							for xc in depths:
								axs[ichan].axvline(x=xc, color='k', linestyle='--', alpha=0.2)
						 
					plt.suptitle(titleLab, y=0.96,x=0.51, size=16, fontweight='bold')
					fig.text(0.51, 0.03, 'Depth (mm)', ha='center', size=14, fontweight='bold')
					fig.text(0.035, 0.5, featureLabels[numFeatures[ifeatures]].values[0], va='center', rotation='vertical', size=14, fontweight='bold')
					
					if showOnly == True:
						plt.show()
					else:
						figure = plt.gcf() # get current figure
						figure.set_size_inches(12, 8)
						if timeWindow==True:
							filepath = outputDir + fileName + '.png'
						else:
							filepath = outputDir + fileName + '.png'
						plt.savefig(filepath, dpi=100)   # save the figure to file
						plt.close('all')
					
					print('Finished subject', str(subList[isub]), numSides[iside], 'side', 'feature:', featureLabels[numFeatures[ifeatures]].values[0])

def extract_raw_nwbFile(file_name, trimData, FilterData):
	
	patientDF = pd.DataFrame([])
	subject = int("".join([x for x in h5py.File(file_name, 'r+').get('/identifier').value.split('_')[0] if x.isdigit()]))
	chans = list(set(h5py.File(file_name, 'r+').get('/intervals/trials/channel').value))    
	with open(file_name.replace('.nwb', '.json')) as side_file:
		sidecar = json.load(side_file)
	Fs = sidecar['SamplingFrequency']
	
	for ichan in chans:
		channelIdx = h5py.File(file_name, 'r+').get('/intervals/trials/channel').value == ichan
		startTime = h5py.File(file_name, 'r+').get('/intervals/trials/start_time').value[channelIdx]
		endTime = h5py.File(file_name, 'r+').get('/intervals/trials/stop_time').value[channelIdx]
		depths = [float(x) for x in h5py.File(file_name, 'r+').get('/intervals/trials/depth').value[channelIdx]]
		dataset = h5py.File(file_name, 'r+').get('/acquisition/'+ ichan +'/data').value
		
		for idx, idepth in enumerate(depths):
			tempData = dataset[int(startTime[idx]):int(endTime[idx])]
			if FilterData:
				tempData = butterBandpass(tempData, lowcut = 400, highcut = 6000, fs = Fs, order = 4)
			rowDF = [{'subject': subject, 'side': h5py.File(file_name, 'r+').get('/session_description').value.split('_')[0], 
					  'channel': ichan, 'chanChosen': np.nan, 'depth': idepth, 'rawData': tempData}]
			patientDF = pd.concat([patientDF, pd.DataFrame(rowDF)], axis = 0)
	
	if trimData == True:
		datasetLength = int(5*np.floor(float(min([len(x) for x in patientDF['rawData']])/Fs)/5))*Fs
		patientDF['rawData'] = [x[:int(datasetLength)] for x in patientDF['rawData']]
		
	return patientDF
#x = filen
#isub = 0
#iside = 0
#ichan = 0
def plotRaw(x, showOnly, processedFolder, Fs, trimData, FilterData):
	
	channelLabels = {1:"Center", 2:"Anterior", 3:"Posterior", 4:"Medial", 5:"Lateral"}
	if not isinstance(x, pd.DataFrame):
		if x.endswith('.nwb'):
			x = extract_raw_nwbFile(x, trimData, FilterData)
			subList = np.unique(x['subject'])
	else:
		subList = np.unique(x['subject'])
	
	for isub in range(len(subList)):
		numSides = np.unique(x[(x['subject']==subList[isub])]['side'])
	
		for iside in range(len(numSides)):
			outputDir = '\\'.join([processedFolder, 'sub-P' + str(subList[isub]).zfill(3), 'rawData', numSides[iside]])
			if not os.path.exists(outputDir):
				os.makedirs(outputDir)
			
			numChans = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['channel'])
			colnames = x.columns.values.tolist()
			if np.isnan(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['chanChosen']).any():
				chanSel = np.nan
			else:
				chanSel = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['chanChosen'])
				
			for ichan in range(len(numChans)):
				if 'labels' in colnames:
					labelsPresent = True
					labels = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['labels'])
				else:
					labelsPresent = False
				if labelsPresent:
					rawData = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['rawData'])
					feature = np.empty((0, len(np.frombuffer(rawData[1,]))))
				else:
					rawData = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['rawData'])
					feature = np.empty((0, len(rawData[1,])))
				
				for idepth in range(len(rawData)):
					if labelsPresent:
						tempdat = np.frombuffer(rawData[idepth,])
						tempdat = butterBandpass(tempdat, lowcut = 500, highcut = 5000, fs = Fs, order = 5)
						feature = np.append(feature, [np.transpose(tempdat)], axis=0)
					else:
						tempdat = rawData[idepth,]
						tempdat = butterBandpass(tempdat, lowcut = 500, highcut = 5000, fs = Fs, order = 5)
						feature = np.append(feature, [np.transpose(tempdat)], axis=0)
				
				nDepths = len(feature)
				yshift = 120
				depths = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['depth'])
				fig, ax = plt.subplots()
				ax.plot(feature.T + yshift * np.arange(0,nDepths,1), color='black', linewidth=0.2)
				ax.yaxis.set_ticks(yshift * np.arange(0,nDepths,1))
				ax.yaxis.set_ticklabels(['{:.2f}'.format(x) for x in depths])
				ax.xaxis.set_ticks(np.arange(0,len(feature.T)+1,(len(feature.T)/5)))
				start, end = ax.get_xlim()
				xTickLabs = np.arange(0, len(feature.T)+1, len(feature.T)/5)/Fs
				ax.xaxis.set_ticklabels(['{:.2f}'.format(x) for x in xTickLabs])
				ax.set_ylim(-yshift,(nDepths*yshift))
				ax.set_xlim(0,len(feature.T))
				
				if labelsPresent:
					if np.size(np.where(labels==1)) != 0:
						inDepth  = np.min(np.where(labels==1))*yshift
						outDepth = np.max(np.where(labels==1))*yshift
						ax.axhline(inDepth, color='green', linewidth=2)
						ax.axhline(outDepth, color='red', linewidth=2)
				
				plt.gca().invert_yaxis()
				
				if isinstance(numChans[ichan], str):
					channel = numChans[ichan]
				else:
					channel = channelLabels.get(numChans[ichan])
					
				if numChans[ichan] == chanSel:
					plt.title('Sub-' + str(subList[isub]).zfill(3) + ' ' + numSides[iside] + ' Side: ' + channel + " Trajectory", size=14, fontweight="bold", color = 'red')
				else:
					plt.title('Sub-' + str(subList[isub]).zfill(3) + ' ' + numSides[iside] + ' Side: ' + channel + " Trajectory", size=14, fontweight="bold")
				
				plt.xlabel("Time (sec)", size=14, fontweight='bold')
				plt.ylabel("Depth (mm)", size=14, fontweight='bold')
				
				fileName = 'sub-P' + str(subList[isub]).zfill(3) + '_side-' + numSides[iside] + '_channel-' + channel + '-rawData'
				figure = plt.gcf() # get current figure
				figure.set_size_inches(20, 12)
				filepath = os.path.join(outputDir, fileName + '.png')
				plt.savefig(filepath, dpi=100)   # save the figure to file
				plt.close()
				
				print('Finished subject', str(subList[isub]), numSides[iside], 'side', 'Raw Data', 'for channel', str(numChans[ichan]))
				
def plotRawBenGun(x, showOnly, processedFolder, Fs, trimData, FilterData):
	
	channelLabels = {1:"Center", 2:"Anterior", 3:"Posterior", 4:"Medial", 5:"Lateral"}
	if not isinstance(x, pd.DataFrame):
		if x.endswith('.nwb'):
			x = extract_raw_nwbFile(x, trimData, FilterData)
			subList = np.unique(x['subject'])
	else:
		subList = np.unique(x['subject'])
			
	for isub in range(len(subList)):
		numSides = np.unique(x[(x['subject']==subList[isub])]['side'])
	
		for iside in range(len(numSides)):
			outputDir = '\\'.join([processedFolder, 'sub-P' + str(subList[isub]).zfill(3), 'rawData', numSides[iside]])
			if not os.path.exists(outputDir):
				os.makedirs(outputDir)
				
			numChans = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['channel'])
			colnames = x.columns.values.tolist()
			if np.isnan(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['chanChosen']).any():
				chanSel = np.nan
			else:
				chanSel = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['chanChosen'])
			if numSides[iside] == 'left':
				axPosition = {1:['5'], 2:['2'], 3:['8'], 4:['6'], 5:['4']}
			else:
				axPosition = {1:['5'], 2:['2'], 3:['8'], 4:['4'], 5:['6']}
			titleLab  = 'Sub-' + str(subList[isub]).zfill(3) + ' ' + numSides[iside] + ' Side: Ben\'s Gun'
			fig = plt.figure()
			for ichan in range(len(numChans)):
				if 'labels' in colnames:
					labelsPresent = True
					labels = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['labels'])
				else:
					labelsPresent = False
				if labelsPresent:
					rawData = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['rawData'])
					feature = np.empty((0, len(np.frombuffer(rawData[1,]))))
				else:
					rawData = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['rawData'])
					feature = np.empty((0, len(rawData[1,])))
				
				for idepth in range(len(rawData)):
					if labelsPresent:
						tempdat = np.frombuffer(rawData[idepth,])
						tempdat = butterBandpass(tempdat, lowcut = 500, highcut = 5000, fs = Fs, order = 5)
						feature = np.append(feature, [np.transpose(tempdat)], axis=0)
					else:
						tempdat = rawData[idepth,]
						tempdat = butterBandpass(tempdat, lowcut = 500, highcut = 5000, fs = Fs, order = 5)
						feature = np.append(feature, [np.transpose(tempdat)], axis=0)
				
				if isinstance(numChans[ichan],str):
					chanPosition = [x[0] for x in list(channelLabels.items()) if numChans[ichan] in x[1]][0]
					channel = numChans[ichan]
				else:
					chanPosition = numChans[ichan]
					channel = channelLabels.get(numChans[ichan])
					
				subPosi = [int(x) for x in axPosition.get(chanPosition)][0]
				nDepths = len(feature)
				yshift = 120
				depths = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['depth'])
				
				ax = plt.subplot(3, 3, subPosi)
				ax.plot(feature.T + yshift * np.arange(0,nDepths,1), color='black', linewidth=0.2)
				ax.yaxis.set_ticks(yshift * np.arange(0,nDepths,1))
				ax.yaxis.set_ticklabels(['{:.2f}'.format(x) for x in depths])
				ax.xaxis.set_ticks(np.arange(0,len(feature.T)+1,(len(feature.T)/5)))
				xTickLabs = np.arange(0, len(feature.T)+1, len(feature.T)/5)/Fs
				ax.xaxis.set_ticklabels(['{:.2f}'.format(x) for x in xTickLabs])
				ax.set_ylim(-yshift,(nDepths*yshift))
				ax.set_xlim(0,len(feature.T))
				
				for label in ax.yaxis.get_ticklabels()[::2]:
					label.set_visible(False)
				plt.gca().invert_yaxis()
				
				if numChans[ichan] == chanSel:
					ax.annotate(channel, xy=(0.42,1.01), xycoords='axes fraction', fontsize=10, fontweight='bold', color = 'red')
				else:
					ax.annotate(channel, xy=(0.42,1.01), xycoords='axes fraction', fontsize=10, fontweight='bold')
				
				if labelsPresent:
					if np.size(np.where(labels==1)) != 0:
						inDepth  = np.min(np.where(labels==1))*yshift
						outDepth = np.max(np.where(labels==1))*yshift
						ax.axhline(inDepth, color='green', linewidth=2)
						ax.axhline(outDepth, color='red', linewidth=2)
						
			# Set common labels
			fig.text(0.51, 0.06, 'Time (sec)', ha='center', va='center', size=12, fontweight="bold")
			fig.text(0.08, 0.5, 'Depth (mm)', ha='center', va='center', rotation='vertical', size=12, fontweight="bold")
			plt.suptitle(titleLab, y=0.94,x=0.51, size=16, fontweight='bold')
			
			fileName = 'sub-P' + str(subList[isub]).zfill(3) + '_side-' + numSides[iside] + '-BensGun'
			figure = plt.gcf() # get current figure
			figure.set_size_inches(20, 12, forward=True)
			filepath = os.path.join(outputDir,fileName + '.png')
			plt.savefig(filepath, dpi=100)   # save the figure to file
			plt.close()
			
			print('Finished subject', str(subList[isub]), numSides[iside], 'side', 'Bens Gun.')

def spikeRaster(spikeTimesFin, patient, side, depths, channel, channelChosen, labels):
	fig = plt.figure()
	ax = plt.subplot(1,1,1)
	spikeTimeClean = []
	for trial in range(len(spikeTimesFin)):
		spikeTime = np.where(spikeTimesFin[trial] > 0)[1]
		spikeTime = spikeTime[np.where(np.diff(spikeTime)>1000)]
		plt.vlines(spikeTime,trial,trial+1)
		spikeTimeClean.append(spikeTime)
	ax.yaxis.set_ticks([x+0.5 for x in range(len(depths))])
	ax.yaxis.set_ticklabels(depths)
	ax.xaxis.set_ticks(np.arange(0,spikeTimesFin[0].shape[1]+1,(spikeTimesFin[0].shape[1])/5))
	start, end = ax.get_xlim()
	ax.xaxis.set_ticklabels(np.arange(0, spikeTimesFin[0].shape[1]+1, spikeTimesFin[0].shape[1]/5)/24000)
	ax.set_xlim(0,spikeTimesFin[0].shape[1])
	plt.gca().invert_yaxis()
	plt.xlabel("Time (sec)")
	plt.ylabel("Depth (mm)") 
	
	if channel == channelChosen:
		plt.title('DBS-' + str(patient) + ' ' + side + ' Side: ' + channelLabels.get(channel) + " Trajectory", fontweight='bold', color = 'red')
	else:
		plt.title('DBS-' + str(patient) + ' ' + side + ' Side: ' + channelLabels.get(channel) + " Trajectory", fontweight='bold')
	
	if any(labels==1)==True:
		 plt.axhline(np.min(np.where(labels==1))+0.5, color='g', linestyle='-', linewidth=2)
		 plt.axhline(np.max(np.where(labels==1))+0.5, color='r', linestyle='-', linewidth=2)
	
	return spikeTimeClean

def prep_nwbFile(file_name):
	
	with h5py.File(file_name,  "r") as f:
		data = f['/processing']
		df = {}
		for item in list(data.items()):
			df[item[0]] = f['/processing/'+item[0]].value.flatten()
	
	subject = int("".join([x for x in h5py.File(file_name, 'r+').get('/identifier').value.split('_')[0] if x.isdigit()]))   
	df['channel'] = h5py.File(file_name, 'r+').get('/intervals/trials/channel').value
	df['depth'] = [float(x) for x in h5py.File(file_name, 'r+').get('/intervals/trials/depth').value]
	df['subject'] =  np.repeat(subject,len(df['channel']))
	df['side'] =  np.repeat(h5py.File(file_name, 'r+').get('/session_description').value.split('_')[0], len(df['channel']))
	df['chanChosen'] = np.repeat(np.nan,len(df['channel']))

	return pd.DataFrame(df)

def plotFeatureMaps(x, showOnly, verticalPlots, reducedFeatures, processedFolder, nSubplots):
	channelLabels = {1:"Center", 2:"Anterior", 3:"Posterior", 4:"Medial", 5:"Lateral"}
	if reducedFeatures == True:
		timeLabels = pd.DataFrame([{'mav': 'Mean Absolute \nValue',
									'variance': 'Variance', 
									'rms': 'Root Mean Square', 
									'curveLength': 'Curve Length',
									'ssi': 'Simple Square \nIntegral',
									'power': 'Power',
									'entropy': 'Signal Entropy',
									'tkeoFour': 'Teager-Kaiser \nEnergy - Four'}])
		frequencyLabels = pd.DataFrame([{'meanF': 'Mean Frequency',
									 'freqRatio': 'Frequency Ratio',
									 'AvgPowerMU': 'Bandpass Power \n(500-1000Hz)', 
									 'AvgPowerSU': 'Bandpass Power \n(1000-3000Hz)',
									 'waveletStd': 'STD of Wavlet \nConvolution'}])
		spikeLabels = pd.DataFrame([])
	else:
		timeLabels = pd.DataFrame([{'mav': 'Mean Absolute \nValue',
									'mavSlope': 'Mean Absolute \nValue Slope',
									'variance': 'Variance', 
									'mmav1': 'Mean Absolute \nValue 1', 
									'mmav2': 'Mean Absolute \nValue 2',
									'rms': 'Root Mean Square', 
									'curveLength': 'Curve Length',
									'zeroCross': 'Zero Crossings', 
									'threshold': 'Threshold', 
									'wamp': 'Willison Amplitude', 
									'ssi': 'Simple Square \nIntegral',
									'power': 'Power',
									'entropy': 'Signal Entropy',
									'peaks': 'Peaks - \nNeg and Pos', 
									'tkeoTwo': 'Teager-Kaiser \nEnergy - Two', 
									'tkeoFour': 'Teager-Kaiser \nEnergy - Four',
									'shapeF': 'Shape Factor',
									'kurtosis': 'Kurtosis', 
									'skew': 'Skewness',
									'crestF': 'Crest Factor'}])
		frequencyLabels = pd.DataFrame([{'meanF': 'Mean Frequency',
										 'freqRatio': 'Frequency Ratio',
										 'AvgPowerMU': 'Bandpass Power \n(500-1000Hz)', 
										 'AvgPowerSU': 'Bandpass Power \n(1000-3000Hz)',
										 'waveletStd': 'STD of Wavlet \nConvolution'}])
		spikeLabels = pd.DataFrame([])
#    spikeLabels = pd.DataFrame([{'spikeISI': 'Inter-Spike Interval', 
#                                 'meanISI': 'Mean of ISI', 
#                                 'stdISI': 'STD of ISI', 
#                                 'burstIndex': 'Burst Index', 
#                                 'pauseIndex': 'Pause Index', 
#                                 'pauseRatio': 'Pause Ratio', 
#                                 'spikeDensity': 'Spike Density'}])
	if not isinstance(x, pd.DataFrame):
		if x.endswith('.nwb'):
			x = prep_nwbFile(x)
			subList = np.unique(x['subject'])
	else:
		subList = np.unique(x['subject'])
	
	for isub in range(len(subList)):
		numSides = np.unique(x[(x['subject']==subList[isub])]['side'])
	
		for iside in range(len(numSides)):
			if verticalPlots == True:
				if reducedFeatures == True:
					outputDir = '\\'.join([processedFolder, 'sub-P' + str(subList[isub]).zfill(3), 'activityMaps-VerticalReduced', numSides[iside]])
				else:
					outputDir = '\\'.join([processedFolder, 'sub-P' + str(subList[isub]).zfill(3), 'activityMaps-Vertical', numSides[iside]])
			else:
				if reducedFeatures == True:
					outputDir = '\\'.join([processedFolder, 'sub-P' + str(subList[isub]).zfill(3), 'activityMaps-Reduced', numSides[iside]])
				else:
					outputDir = '\\'.join([processedFolder, 'sub-P' + str(subList[isub]).zfill(3), 'activityMaps', numSides[iside]])
			
			if not os.path.exists(outputDir):
				os.makedirs(outputDir)
	
			numChans = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['channel'])
			colnames = x.columns.values.tolist() 
			if 'labels' in colnames:
				labelsPresent = True
				numFeatures = list(x.drop(['subject','side','channel','depth','labels', 'chanChosen'], axis=1))
			else:
				labelsPresent = False
				numFeatures = list(x.drop(['subject','side','channel','depth', 'chanChosen'], axis=1))
			numTime = list(set(list(timeLabels)).intersection(numFeatures))
			numFreq = list(set(list(frequencyLabels)).intersection(numFeatures))
			numSpike = list(set(list(spikeLabels)).intersection(numFeatures))
			featureDomains = {'Time': numTime, 'Frequency': numFreq,'Spike': numSpike}
			featureDomains.setdefault('Time', []).append(timeLabels)
			featureDomains.setdefault('Frequency', []).append(frequencyLabels)
			featureDomains.setdefault('Spike', []).append(spikeLabels)
			
			for ichan in range(len(numChans)):
				depths = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['depth'])
				if labelsPresent:
					labels = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['labels'])
				if isinstance(numChans[ichan],str):
					channel = numChans[ichan]
				else:
					channel = channelLabels.get(numChans[ichan])
				
				for iDomain in range(3):
					domainName = list(featureDomains.keys())[iDomain]
					numDomain = list(featureDomains.values())[iDomain][:-1]
					featureLabel = list(featureDomains.values())[iDomain][-1]
					
					if len(numDomain)>0:
						numFigs = int(np.floor(len(numDomain)/nSubplots))
						nSubplotsReal = [nSubplots] * numFigs
						if len(numDomain)%nSubplots !=0:
							numFigs += 1
							if not nSubplotsReal:
								nSubplotsReal = [len(numDomain)%nSubplots]
							else:
								nSubplotsReal.append(len(numDomain)%nSubplots)
							
						nStart = 0
						for iplot in range(numFigs):
							if verticalPlots == True:
								fig, axs = plt.subplots(1,nSubplotsReal[iplot], sharex=False, sharey=True)
								fig.subplots_adjust(hspace=0, wspace=0.1)
							else:
								fig, axs = plt.subplots(nSubplotsReal[iplot],1, sharex=True, sharey=False)
								fig.subplots_adjust(hspace=0.1, wspace=0)
							titleLab  = 'Sub-' + str(subList[isub]).zfill(3) + ' ' + numSides[iside] + ' Side: ' + channel + ' Channel - ' + domainName + ' Features #' + str(iplot+1)
							fileName = 'sub-P' + str(subList[isub]).zfill(3) + '_side-' + numSides[iside] + '_channel-' + channel + '-' + domainName + 'Features' + str(iplot+1)
							axCount = 0
							
							nEnd = nStart + nSubplotsReal[iplot]
								
							for ifeatures in range(nStart, nEnd):
								feature = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])][numDomain[ifeatures]])
								feature = (feature - min(feature))/(max(feature)-min(feature))
								muA = np.mean(feature)
								if verticalPlots == True:
									axs[axCount].plot(feature, depths)
									axs[axCount].set_ylim(depths[0,],depths[-1])
									axs[axCount].set_xlabel(featureLabel[numDomain[ifeatures]].values[0], fontsize=10, fontweight='bold')
									axs[axCount].plot([muA,muA], axs[axCount].get_ylim(), ls= 'dashed', c='black')
								else:
									axs[axCount].plot(depths, feature)
									axs[axCount].set_xlim(depths[0,],depths[-1])
									axs[axCount].annotate(featureLabel[numDomain[ifeatures]].values[0], xy=(1.01,0.5), xycoords='axes fraction', fontsize=10, fontweight='bold')
									axs[axCount].plot(axs[axCount].get_xlim(), [muA,muA], ls= 'dashed', c='black')
								if labelsPresent:
									if np.size(np.where(labels==1)) != 0:
										inDepth  = depths[np.min(np.where(labels==1))]
										outDepth = depths[np.max(np.where(labels==1))]
										axs[axCount].axvspan(inDepth, outDepth, color='purple', alpha=0.2)
								
								for xc in depths:
									if verticalPlots == True:
										axs[axCount].axhline(y=xc, color='k', linestyle='--', alpha=0.2)
									else:
										axs[axCount].axvline(x=xc, color='k', linestyle='--', alpha=0.2)
							
								axs[axCount].invert_yaxis()
								if verticalPlots == True and axCount == 0:
									axs[axCount].set_ylabel('Depth (mm)', size=14, fontweight='bold')
								if verticalPlots == True and axCount == (int(np.ceil(nSubplotsReal[iplot]/2))-1):
									if nSubplotsReal[iplot]%2 !=0:
										axs[axCount].annotate('Normalized Units', xy=(0,-.2), xycoords='axes fraction', fontsize=14, fontweight='bold')
									else:
										axs[axCount].annotate('Normalized Units', xy=(0.5,-.2), xycoords='axes fraction', fontsize=14, fontweight='bold')
										
								if verticalPlots == False and axCount == (int(np.ceil(nSubplotsReal[iplot]/2))-1):
									if nSubplotsReal[iplot]%2 !=0:
										axs[axCount].set_ylabel('Normalized Units', size=14, fontweight='bold')
									else:
										axs[axCount].set_ylabel('Normalized Units', size=14, fontweight='bold')
										axs[axCount].yaxis.set_label_coords(-.05,0)
								
								axCount +=1
							
							if verticalPlots == True:
								axs[(axCount-1)].yaxis.set_ticks(depths)
								axFormaty(axs[(axCount-1)])
								plt.suptitle(titleLab, y=0.94,x=0.51, size=16, fontweight='bold')
								plt.subplots_adjust(bottom=0.20)
								if nSubplotsReal[iplot] == 2:
									plt.subplots_adjust(left=0.35)
									plt.subplots_adjust(right=0.65)
								elif nSubplotsReal[iplot] == 3:
									plt.subplots_adjust(left=0.27)
									plt.subplots_adjust(right=0.73)
								elif nSubplotsReal[iplot] == 4:
									plt.subplots_adjust(left=0.19)
									plt.subplots_adjust(right=0.81)
									
							else:
								start, end = axs[axCount-1].get_xlim()
								axs[axCount-1].xaxis.set_ticks(np.linspace(depths[0], depths[-1], len(depths)))
								axs[axCount-1].xaxis.set_ticklabels(['{:.2f}'.format(x) for x in depths], rotation=45)
								plt.subplots_adjust(right=0.80)
								
								if nSubplotsReal[iplot] == 2:
									plt.subplots_adjust(bottom=0.57)
								elif nSubplotsReal[iplot] == 3:
									plt.subplots_adjust(bottom=0.415)
								elif nSubplotsReal[iplot] == 4:
									plt.subplots_adjust(bottom=0.265)
								
								plt.suptitle(titleLab, y=0.96,x=0.46, size=16, fontweight='bold')
								plt.xlabel('Depth (mm)', size=14, fontweight='bold')
							
							nStart += nSubplotsReal[iplot]
							
							if showOnly == True:
								plt.show()
							else:
								figure = plt.gcf() # get current figure
								figure.set_size_inches(12, 8)
								filepath = os.path.join(outputDir , fileName + '.png')
								plt.savefig(filepath, dpi=100)   # save the figure to file
								plt.close('all')
								
				print('Finished subject', str(subList[isub]), numSides[iside], 'side', 'channel', numChans[ichan])   

def plotFeatureMaps_gui(x, verticalPlots, processedFolder, nSubplots):
	channelLabels = {1:"Center", 2:"Anterior", 3:"Posterior", 4:"Medial", 5:"Lateral"}
	timeLabels = pd.DataFrame([{'mav': 'Mean Absolute \nValue',
								'rms': 'Root Mean Square', 
								'curveLength': 'Curve Length',
								'power': 'Power',
								'entropy': 'Signal Entropy',
								'tkeoFour': 'Teager-Kaiser \nEnergy - Four'}])
	frequencyLabels = pd.DataFrame([])
	spikeLabels = pd.DataFrame([])
#    spikeLabels = pd.DataFrame([{'spikeISI': 'Inter-Spike Interval', 
#                                 'meanISI': 'Mean of ISI', 
#                                 'stdISI': 'STD of ISI', 
#                                 'burstIndex': 'Burst Index', 
#                                 'pauseIndex': 'Pause Index', 
#                                 'pauseRatio': 'Pause Ratio', 
#                                 'spikeDensity': 'Spike Density'}])
	
	subList = np.unique(x['subject'])
	rowFinal = []
	plotFinal = []
	for isub in range(len(subList)):
		plots = {}
		if verticalPlots == True:
			plots['outputDir'] = processedFolder + '/sub-' + str(subList[isub]) + '/activityMaps-Vertical/'
		else:
			plots['outputDir'] =  processedFolder + '/sub-' + str(subList[isub]) + '/activityMaps/'
		
		numSides = np.unique(x[(x['subject']==subList[isub])]['side'])
		
		plotFinal.append(plots)
		for iside in range(len(numSides)):
			numChans = np.unique(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside])]['channel'])
			colnames = x.columns.values.tolist() 
			if 'labels' in colnames:
				labelsPresent = True
				numFeatures = list(x.drop(['subject','side','channel','depth','labels', 'chanChosen'], axis=1))
			else:
				labelsPresent = False
				numFeatures = list(x.drop(['subject','side','channel','depth', 'chanChosen'], axis=1))
			
			numTime = list(set(list(timeLabels)).intersection(numFeatures))
			numFreq = list(set(list(frequencyLabels)).intersection(numFeatures))
			numSpike = list(set(list(spikeLabels)).intersection(numFeatures))
			featureDomains = {'Time': numTime, 'Frequency': numFreq,'Spike': numSpike}
			featureDomains.setdefault('Time', []).append(timeLabels)
			featureDomains.setdefault('Frequency', []).append(frequencyLabels)
			featureDomains.setdefault('Spike', []).append(spikeLabels)
			
			for ichan in range(len(numChans)):
				depths = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['depth'])
				if labelsPresent:
					labels = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])]['labels'])
				channel = channelLabels.get(numChans[ichan])
				for iDomain in range(3):
					domainName = list(featureDomains.keys())[iDomain]
					numDomain = list(featureDomains.values())[iDomain][:-1]
					featureLabel = list(featureDomains.values())[iDomain][-1]
					
					if len(numDomain)>0:
						numFigs = int(np.floor(len(numDomain)/nSubplots))
						nSubplotsReal = [nSubplots] * numFigs
						if len(numDomain)%nSubplots !=0:
							numFigs += 1
							if not nSubplotsReal:
								nSubplotsReal = [len(numDomain)%nSubplots]
							else:
								nSubplotsReal.append(len(numDomain)%nSubplots)
							
						nStart = 0
						for iplot in range(numFigs):
							
#                            if verticalPlots == True:
#                                fig, axs = plt.subplots(1,nSubplotsReal[iplot], sharex=False, sharey=True)
#                                fig.subplots_adjust(hspace=0, wspace=0.1)
#                            else:
#                                fig, axs = plt.subplots(nSubplotsReal[iplot],1, sharex=True, sharey=False)
#                                fig.subplots_adjust(hspace=0.1, wspace=0)
							titleLab  = 'Sub-' + str(subList[isub]) + ' ' + numSides[iside] + ' Side: ' + channel + ' Channel - ' + domainName + ' Features #' + str(iplot+1)
							fileName = 'sub-' + str(subList[isub]) + '_side-' + numSides[iside] + '_channel-' + channel + '-' + domainName + 'Features' + str(iplot+1)
							
							axCount = 0
							nEnd = nStart + nSubplotsReal[iplot]
							for ifeatures in range(nStart, nEnd):
								row = {}
								row['subject'] = str(subList[isub])
								row['side'] = numSides[iside]
								row['channel'] = channel
								row['domain'] = domainName
								row['plotTitle'] = titleLab
								row['fileName'] = fileName

								feature = np.array(x[(x['subject']==subList[isub]) & (x['side'] == numSides[iside]) & (x['channel'] == numChans[ichan])][numDomain[ifeatures]])
								feature = (feature - min(feature))/(max(feature)-min(feature))
								featureMean = np.mean(feature)
								
								if verticalPlots == True:
									row['plot'] = ['plot',feature, depths]
									row['featureMean'] = ['plot', [featureMean,featureMean], 'get_ylim()', 'dashed', 'black']
									row['depthLim'] = ['set_ylim', [depths[0,],depths[-1]]]
									row['featureLabel'] = ['set_xlabel', featureLabel[numDomain[ifeatures]].values[0], 10, 'bold']
								else:
									row['plot'] = ['plot',depths, feature]
									row['featureMean'] = ['plot', 'get_xlim()', [featureMean,featureMean], 'dashed', 'black']
									row['depthLim'] = ['set_xlim', [depths[0,],depths[-1]]]
									row['featureLabel'] = ['annotate', featureLabel[numDomain[ifeatures]].values[0], [1.01,0.5], 'axes fraction', 10, 'bold']
								if labelsPresent:
									if np.size(np.where(labels==1)) != 0:
										inDepth  = depths[np.min(np.where(labels==1))]
										outDepth = depths[np.max(np.where(labels==1))]
										row['labels'] = [inDepth, outDepth]
										if verticalPlots == True:
											row['labelsType'] = ['axhspan', [inDepth, outDepth], 'purple', 0.2]
										else:
											row['labelsType'] = ['axvspan', [inDepth, outDepth], 'purple', 0.2]
								for xc in depths:
									if verticalPlots == True:
										row['depthMark'] = ['axhline', 'y', 'k', 0.2, '--']
									else:
										row['depthMark'] = ['axvline', 'x', 'k', 0.2, '--']
							
								if verticalPlots == True and axCount == 0:
									row['yLabel'] = ['set_ylabel', 'Depth (mm)', 14, 'bold']
								if verticalPlots == True and axCount == (int(np.ceil(nSubplotsReal[iplot]/2))-1):
									if nSubplotsReal[iplot]%2 !=0:
										row['yLabel'] = ['annotate', 'Normalized Units', [0,-.2], 'axes fraction', 14, 'bold']
									else:
										row['yLabel'] = ['annotate', 'Normalized Units', [0.5,-.2], 'axes fraction', 14, 'bold']
										
								if verticalPlots == False and axCount == (int(np.ceil(nSubplotsReal[iplot]/2))-1):
									if nSubplotsReal[iplot]%2 !=0:
										row['yLabel'] = ['set_ylabel', 'Normalized Units', 14, 'bold']
									else:
										row['yLabel'] = ['set_ylabel', 'Normalized Units', [-.05,0], 'yaxis.set_label_coords', 14, 'bold']                                
								
								rowFinal.append(dict(zip(row.keys(), row.values())))
								
								axCount +=1
							
#                            if verticalPlots == True:
#                                axs[(axCount-1)].yaxis.set_ticks(depths)
#                                axFormaty(axs[(axCount-1)])
#                                plt.suptitle(titleLab, y=0.94,x=0.51, size=16, fontweight='bold')
#                                plt.subplots_adjust(bottom=0.20)
#                                if nSubplotsReal[iplot] == 2:
#                                    plt.subplots_adjust(left=0.35)
#                                    plt.subplots_adjust(right=0.65)
#                                elif nSubplotsReal[iplot] == 3:
#                                    plt.subplots_adjust(left=0.27)
#                                    plt.subplots_adjust(right=0.73)
#                                elif nSubplotsReal[iplot] == 4:
#                                    plt.subplots_adjust(left=0.19)
#                                    plt.subplots_adjust(right=0.81)
#                                    
#                            else:
#                                start, end = axs[axCount-1].get_xlim()
#                                axs[axCount-1].xaxis.set_ticks(np.linspace(depths[0], depths[-1], len(depths)))
#                                axs[axCount-1].xaxis.set_ticklabels(['{:.2f}'.format(x) for x in depths], rotation=45)
#                                plt.subplots_adjust(right=0.80)
#                                
#                                if nSubplotsReal[iplot] == 2:
#                                    plt.subplots_adjust(bottom=0.57)
#                                elif nSubplotsReal[iplot] == 3:
#                                    plt.subplots_adjust(bottom=0.415)
#                                elif nSubplotsReal[iplot] == 4:
#                                    plt.subplots_adjust(bottom=0.265)
#                                
#                                plt.suptitle(titleLab, y=0.96,x=0.46, size=16, fontweight='bold')
#                                plt.xlabel('Depth (mm)', size=14, fontweight='bold')
							
							nStart += nSubplotsReal[iplot]
							
#                            
								
				print('Finished subject', str(subList[isub]), numSides[iside], 'side', 'channel', numChans[ichan])  
				
	return rowFinal

def plotFFT(data, Fs, facet=False, freqMin=1, freqMax=5000, yMin=None, yMax=None):
	"""
	Create the x-axis and plot the FFT of data.
	
	Parameters
	----------
		data: array-like
			Data containing the frequency series to plot. Each column is an
			electrode.
		facet: bool, default to False
			If True, each electrode will be plotted on a different facet.
		freqMin: float, default to None
			Minimum frequency (x-axis) to show on the plot.
		freqMax: float, default to None
			Maximum frequency (x-axis) to show on the plot.
		yMin: float, default to None
			Minimum value (y-axis) to show on the plot.
		yMax: float, default to None
			Maximum value (y-axis) to show on the plot.
		fs: float
			Sampling frequency of data in Hz.
	
	Returns
	-------
		fig: instance of matplotlib.figure.Figure
			The figure of the FFT.
	"""

	tf, fftData = computeFFT(data, Fs)
	yMax = np.mean(fftData) + (np.std(fftData)*12)
	
	plt.figure()
	plt.plot(tf, fftData, linewidth=0.5)
	
	if (freqMin is not None):
		plt.xlim(left=freqMin)
	if (freqMax is not None):
		plt.xlim(right=freqMax)
	if (yMin is not None):
		plt.ylim(bottom=yMin)
	if (yMax is not None):
		plt.ylim(top=yMax)
			
	plt.xlabel('frequency (Hz)')

##############################################################################
#                               SPIKE SORTING                                #
##############################################################################
def spikeSorting(outputChan, combinatoDir, optimize):

	changeDir = 'cd ' + outputChan
	extract = 'python ' + combinatoDir + '/css-extract --matfile'
	cluster = 'python '+ combinatoDir + '/css-simple-clustering {} --datafile'
	
	mat_files = [f for f in os.listdir(changeDir[3:]) if f.endswith('.mat')]
	mat_files = sorted_nicely(mat_files)

	for ifile in range(len(mat_files)):
		filen = mat_files[ifile]
		newData = filen[:-4] + '/' + 'data_' + filen[:-4] + '.h5'
		
		#--- Extract
		command = extract + ' ' + filen
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE , shell=True, cwd=changeDir[3:])
		stdout = process.communicate()[0]
		
		if optimize == True:
			options = {'MaxClustersPerTemp': 7,
				   'RecursiveDepth': 2,
				   'MinInputSizeRecluster': 1000,
				   'MaxDistMatchGrouping': 1.6,
				   'MarkArtifactClasses': False,
				   'RecheckArtifacts': False}
			localOp = changeDir[3:] + '/' + mat_files[ifile][:-4] + '/local_options'
			np.save(localOp, options)
			os.rename(localOp + '.npy', localOp + '.py')
			
			commandNeg = cluster.format('--neg') + ' ' + newData + ' --label optimized'
			commandPos = cluster.format('') + ' ' + newData + ' --label optimized'
		else:
			commandNeg = cluster.format('--neg') + ' ' + newData
			commandPos = cluster.format('') + ' ' + newData
			
		#--- Sort Negative
		process = subprocess.Popen(commandNeg.split(), stdout=subprocess.PIPE , shell=True, cwd=changeDir[3:])
		stdout = process.communicate()[0]
		
		#--- Sort Positive
		process = subprocess.Popen(commandPos.split(), stdout=subprocess.PIPE , shell=True, cwd=changeDir[3:])
		stdout = process.communicate()[0]
		
		print("Done extracting/clustering file {} of {}: ".format(str(ifile +1), str(len(mat_files))), mat_files[ifile])

def spikeSortResults(outputChan, removeArtifacts, detectionTypes):
	spikeTimesPos = []
	spikesPos = []
	spikesNeg = []
	spikeTimesNeg = []
	
	mat_files = [f for f in os.listdir(outputChan) if f.endswith('.mat')]
	mat_files = sorted_nicely(mat_files)
	
	for ifile in range(len(mat_files)):
		filen = mat_files[ifile][:-4]
		for idetect in range(len(detectionTypes)):
			checkClass = outputChan + '/' + filen + '/sort_' + detectionTypes[idetect]  + '_simple/sort_cat.h5'
			spikesTemp = []
			spikeTimeTemp = []
			
			if os.path.isfile(checkClass):
				fid = tables.open_file(checkClass, 'r')
				allClass = np.unique(fid.get_node('/classes')[:])
				Types = fid.get_node('/artifacts')[:]
				
				if any(allClass!=0):
					allClass = allClass[allClass>0]
					
					if removeArtifacts == True:
						allClassFinal = []
						for iclass in range(len(allClass)):
							if Types[allClass[iclass],1] !=1:
								allClassFinal.append(allClass[iclass])
						
						if len(allClassFinal) > 0:
							classes = fid.get_node('/classes')[:]
							matches = fid.get_node('/matches')[:]
							fid.close()
							h5File = outputChan + '/' + filen + '/data_' + filen + '.h5'
							fid = tables.open_file(h5File, 'r')
							spk = fid.get_node('/' + detectionTypes[idetect] + '/spikes')[:, :]
							spk = spk[(classes>0) & (matches>0),:]
							time = fid.get_node('/' + detectionTypes[idetect] + '/times')[:]
							spikesTemp.append(np.column_stack((classes[(classes>0) & (matches>0)], spk)))
							spikeTimeTemp.append(time[(classes>0) & (matches>0)])
							fid.close()
						else:
							spikesTemp = []
							spikeTimeTemp = []
					else:
						classes = fid.get_node('/classes')[:]
#                        matches = fid.get_node('/matches')[:]
						h5File = outputChan + '/' + filen + '/data_' + filen + '.h5'
						fid = tables.open_file(h5File, 'r')
						spk = fid.get_node('/' + detectionTypes[idetect] + '/spikes')[:, :]
#                        spk = spk[(classes>0) & (matches>0),:]
						spk = spk[(classes>0),:]
						time = fid.get_node('/' + detectionTypes[idetect] + '/times')[:]
#                        spikesTemp.append(np.column_stack((classes[(classes>0) & (matches>0)], spk)))
#                        spikeTimeTemp.append(time[(classes>0) & (matches>0)])
						spikesTemp.append(np.column_stack((classes[(classes>0)], spk)))
						spikeTimeTemp.append(time[(classes>0)])
						fid.close()
				else:
					spikesTemp = []
					spikeTimeTemp = []
					fid.close()
			else:
				spikesTemp = []
				spikeTimeTemp = []
				
			if 'pos' in detectionTypes[idetect]:
				if len(spikesTemp) > 0:
					spikesPos.append(spikesTemp[0])
					spikeTimesPos.append(spikeTimeTemp[0])
				else:
					spikesPos.append(spikesTemp)
					spikeTimesPos.append(spikeTimeTemp)
			else:
				if len(spikesTemp) > 0:
					spikesNeg.append(spikesTemp[0])
					spikeTimesNeg.append(spikeTimeTemp[0])
				else:
					spikesNeg.append(spikesTemp)
					spikeTimesNeg.append(spikeTimeTemp)    
	spikeResults = []
	spikeResults = [{'PositiveSpikes': spikesPos, 'PositiveTimes': spikeTimesPos, 'NegativeSpikes': spikesNeg, 'NegativeTimes': spikeTimesNeg}]
	
	return spikeResults


##############################################################################
#                              SPIKE FEATURES                                #
##############################################################################
def spikeISI(x):
	if len(x)>1:
		xISI = x[1:] - x[:-1]
	else:
		xISI = 0
		
	return xISI

def meanISI(x):
	if len(x)>1:
		x1 = x[1:] - x[:-1]
		xmeanISI = sum(x1)/len(x1)
	else:
		xmeanISI = 0
		
	return xmeanISI

def stdISI(x):
	if len(x)>1:
		x1 = x[1:] - x[:-1]
		xstdISI = np.std(x1)
	else:
		xstdISI = 0
		
	return xstdISI

def burstIndex(x):
	x1 = x[1:] - x[:-1]
	if len(x)>5 & (x1[np.where(x1>10)[0]].shape[0]) > 0:
		xburstIndex = (x1[np.where(x1<10)[0]].shape[0])/(x1[np.where(x1>10)[0]].shape[0])
	else:
		xburstIndex = 0
		
	return xburstIndex

def pauseIndex(x):
	x1 = x[1:] - x[:-1]
	if len(x)>5 & (x1[np.where(x1<50)[0]].shape[0]) > 0:
		xpauseIndex = (x1[np.where(x1>50)[0]].shape[0])/(x1[np.where(x1<50)[0]].shape[0])
	else:
		xpauseIndex = 0
		
	return xpauseIndex

def pauseRatio(x):
	x1 = x[1:] - x[:-1]
	if len(x)>5 & int(sum(x1[np.where(x1<50)[0]])) > 0:
		xpauseRatio = sum(x1[np.where(x1>50)[0]])/sum(x1[np.where(x1<50)[0]])
	else:
		xpauseRatio = 0
		
	return xpauseRatio

def spikeDensity(x):
	if len(x)>1:
		xspikeDensity = len(x)
	else:
		xspikeDensity = 0
		
	return xspikeDensity
