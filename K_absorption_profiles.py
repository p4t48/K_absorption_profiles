import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import find_peaks_cwt
import sys
import step_detect as sd


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

    def AbsorptionSignal(self, N, channel=0, plot=False):

        boundaries = self.TriggerAbsorption(N, channel, plot)
        absStart = boundaries[0]
        absStop = boundaries[1]

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

        # Finding where to cut the data because of mode-hops or other
        jumps = sd.mz_fwt(data, n=2)
        jumps /= np.abs(jumps).max()
        jumpLocations = np.where(jumps >= 1e-4)[0] # 1e-4 seems to properly discriminate mode-hops
        print(jumpLocations)
        # Cutting data from the first mode-hop on
        dataCut = data[jumpLocations.max()+100:]
        maximaLocations, minimaLocations = self.PeakDet(data, 0.1)
        
        if plot == True:
            
            # Plotting all relevant points and the data
            xVals = np.linspace(0,1,data.size)
            xValsCut = xVals[jumpLocations.max()+100:]
            xValsMinima = np.take(xVals, minimaLocations)            
            xValsJumps = np.take(xVals, jumpLocations)
            jumpVals = np.take(data, jumpLocations)
            print(jumpVals)
            minVals = np.take(data, minimaLocations)
            plt.plot(xVals, data, label='original data')
            plt.plot(xVals, jumps, label='detected jumps')
            plt.plot(xValsCut, dataCut, 'k', label='cut data')
            plt.plot(xValsJumps, jumpVals, 'ro', label='jump locations')
            plt.plot(xValsMinima, minVals, 'ks', label='minima locations')
            plt.legend()
            plt.show()

        return dataCut
    
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
