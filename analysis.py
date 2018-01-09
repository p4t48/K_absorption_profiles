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

# Data files location        
dataFiles = glob.glob("../20171221/*_5_probe")
print(dataFiles)

# Work on the data
an = KAbsProfiles(dataFiles[0], samplingRate, bits, channelLayout, channelRanges, amplifierGains)
#an.PlotChannelData(1)
#an.AbsorptionSignal(1,2,True)
#print(an.AbsorptionParameters(1,2))
#print(an.FitAbsorptionProfile(1,2,1))
an.AnalyseNProfiles(5,2)
