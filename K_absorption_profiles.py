import numpy as np
import matplotlib.pyplot as plt
import glob


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

        self.channel1 = self.data[channelLayout['ch1']-1::4] * channel1Norm            
        self.channel2 = self.data[channelLayout['ch2']-1::4] * channel2Norm
        self.channel3 = self.data[channelLayout['ch3']-1::4] * channel3Norm
        self.channel4 = self.data[channelLayout['ch4']-1::4] * channel4Norm

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

        plt.plot(xVals, data)
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

        
dataFiles = glob.glob("../20171221/*_5_*")
print(dataFiles)
        
an = KAbsProfiles(dataFiles[0], samplingRate, bits, channelLayout, channelRanges, amplifierGains)

an.PlotChannelData(4)
