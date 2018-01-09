import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AbsorptionResults:

    def __init__(self, resultFiles):
        self.files = glob.glob(resultFiles)
        print(self.files)

    def PlotData(self):

        probePowers, pumpPowers = [], []
        probeAbsorption, pumpAbsorption = [], []
        probeAbsError, pumpAbsError = [], []
        
        for dataFile in self.files:
            print(dataFile)
            df = pd.read_csv(dataFile, sep="\t")
            df.reindex(df.index.drop(1))
            power = df["Laser Powers (V)"]
            absorption = df["Peak Height (V)"]
            if dataFile.find("_probe") == -1:
                pumpPowers.append(np.mean(power))
                pumpAbsorption.append(np.mean(absorption))
                pumpAbsError.append(np.std(absorption)/np.sqrt(len(absorption)))
            else:
                probePowers.append(np.mean(power))
                probeAbsorption.append(np.mean(absorption))
                probeAbsError.append(np.std(absorption)/np.sqrt(len(absorption)))

        print(len(probeAbsError))
        print(len(probeAbsorption))
        plt.xlabel(r"Laser power, P ($\mu$A)", size=26)
        plt.ylabel("Absorption depth (V)", size=26)
        plt.errorbar(pumpPowers, pumpAbsorption, yerr=pumpAbsError, fmt='o', color='b', label="Pump beam")
        plt.errorbar(probePowers, probeAbsorption, yerr=probeAbsError, fmt='o', color='r', label="Probe beam")
        plt.legend()
        plt.tight_layout()
        plt.show()
