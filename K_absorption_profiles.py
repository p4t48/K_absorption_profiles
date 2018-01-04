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


dataFiles = glob.glob("../20171221/*_5_*")
print(dataFiles)
        
