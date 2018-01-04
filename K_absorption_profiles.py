import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import find_peaks_cwt
import sys


class KAbsProfiles:

    def __init__(self, dataFile, samplingRate, bits, channelLayout, channelRanges, amplifierGains):
        self.dataFile = dataFile
        self.samplingRate = samplingRate
        self.amplifierGains = amplifierGains

        # Depending on the data format, get signal in volts
        f = open("%s" % self.dataFile, "r")
        if bits == 16:
            self.data = np.fromfile(f, dtype=np.int16)
            channel1Norm = channelRanges['ch1'] / 2**15 # To get voltages from 16 bit integer
            channel2Norm = channelRanges['ch2'] / 2**15 
            channel3Norm = channelRanges['ch3'] / 2**15 
            channel4Norm = channelRanges['ch4'] / 2**15 
        elif bits == 32:
            self.data = np.fromfile(f, dtype=np.int32)
            channel1Norm = channelRanges['ch1'] / 2**31 # To get voltages from 18 bit integer
            channel2Norm = channelRanges['ch2'] / 2**31 
            channel3Norm = channelRanges['ch3'] / 2**31 
            channel4Norm = channelRanges['ch4'] / 2**31             
        else:
            print("Needs to be either 16 or 32 bit!")

        # Get voltage values for channel data
        self.channel1 = self.data[channelLayout['ch1']-1::4] * channel1Norm            
        self.channel2 = self.data[channelLayout['ch2']-1::4] * channel2Norm
        self.channel3 = self.data[channelLayout['ch3']-1::4] * channel3Norm
        self.channel4 = self.data[channelLayout['ch4']-1::4] * channel4Norm


    def PeakDet(self, data, delta, indices = None):
        """ Function adapted from https://gist.github.com/endolith/250860.
            Returns the maxima and minima with gap delta of an array."""
        
        maxima, minima = [], []
        
        if indices is None:
            indices = np.arange(len(data))

        data = np.asarray(data)
    
        if len(data) != len(indices):
            sys.exit('Input vectors data and indices must have same length')
        
        if not np.isscalar(delta):
            sys.exit('Input argument delta must be a scalar')
        
        if delta <= 0:
            sys.exit('Input argument delta must be positive')
        
        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN
        lookformax = True
    
        for i in np.arange(len(data)):
            this = data[i]
            if this > mx:
                mx = this
                mxpos = indices[i]
            if this < mn:
                mn = this
                mnpos = indices[i]

            if lookformax:
                if this < mx-delta:
                    maxima.append(mxpos)
                    mn = this
                    mnpos = indices[i]
                    lookformax = False
            else:
                if this > mn+delta:
                    minima.append(mnpos)
                    mx = this
                    mxpos = indices[i]
                    lookformax = True

        return np.array(maxima), np.array(minima)

    def TriggerAbsorption(self, N, channel=0, plot=False):
        """ Trigger the Nth absorption profile and plot the data if wanted. """

        maximaLocations, minimaLocations = self.PeakDet(self.channel4, 0.1)
        absStart = minimaLocations[N-1]
        absStop = maximaLocations[N]

        if channel == 1:
            data = self.channel1[absStart:absStop]
        elif channel == 2:
            data = self.channel2[absStart:absStop]
        elif channel == 3:
            data = self.channel3[absStart:absStop]
        elif channel == 4:
            data = self.channel4[absStart:absStop]
        else:
            pass

        if plot == True:
            xVals = np.linspace(0,1,data.size)
            plt.plot(xVals, data)
            plt.show()

        return [absStart, absStop]
    
    def PlotChannelData(self, channel):

        if channel == 1:
            data =  self.channel1
        elif channel == 2:
            data = self.channel2
        elif channel == 3:
            data = self.channel3
        elif channel == 4:
            data = self.channel4

        xVals = np.linspace(0,1,len(data))
        maximaLocations, minimaLocations = self.PeakDet(data, 0.1)
        print(maximaLocations)
        print(minimaLocations)
        
        xValsMax = np.take(xVals, maximaLocations)
        xValsMin = np.take(xVals, minimaLocations)
        maxima = np.take(data, maximaLocations)
        minima = np.take(data, minimaLocations)
        print(maxima.size)
        print(xValsMax.size)
                
        plt.plot(xVals, data)
        plt.plot(xValsMax, maxima, 'ro')
        plt.plot(xValsMin, minima, 'ko')
        plt.show()

#
# Analyse data
#

# Info about input data file
samplingRate = 2*10**4
bits = 16
channelLayout = {'ch1': 1, 'ch2': 2, 'ch3': 3, 'ch4': 4}
channelRanges = {'ch1': 10, 'ch2': 10, 'ch3': 10, 'ch4': 10}
amplifierGains = {'ch1': 10**6, 'ch2': 10**6, 'ch3': 1, 'ch4': 1}

        
dataFiles = glob.glob("../20171222/*_5_*")
print(dataFiles)
        
an = KAbsProfiles(dataFiles[1], samplingRate, bits, channelLayout, channelRanges, amplifierGains)

#an.PlotChannelData(4)
print(an.TriggerAbsorption(5,1,True))
