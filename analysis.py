import glob
from K_absorption_profiles import *

#
# Analyse data
#

# Info about input data file
samplingRate = 2*10**4
bits = 16
channelLayout = {'ch1': 1, 'ch2': 2, 'ch3': 3, 'ch4': 4}
channelRanges = {'ch1': 10, 'ch2': 10, 'ch3': 10, 'ch4': 10}
amplifierGains = {'ch1': 10**6, 'ch2': 10**6, 'ch3': 1, 'ch4': 1}

        
dataFiles = glob.glob("../20171222/*_7_*")
print(dataFiles)
        
an = KAbsProfiles(dataFiles[1], samplingRate, bits, channelLayout, channelRanges, amplifierGains)

#an.PlotChannelData(4)
#print(an.TriggerAbsorption(5,1,True))
print(an.AbsorptionSignal(5,2,True))
