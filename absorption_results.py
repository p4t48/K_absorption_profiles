import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lmfit as lf

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

    def FitData(self):

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
        
        power = lf.models.PowerLawModel()
        pars = power.make_params()
        pars['amplitude'].set(-0.3)
        pars['exponent'].set(-0.6)
        x = np.array(probePowers)
        y = np.array(probeAbsorption)
        arguments = np.argsort(x)
        x = x[arguments]
        y = y[arguments]        

        # Fit data with the standard least squares
        init = power.eval(pars, x=x)
        out = power.fit(y, pars, x=x)
        comps = out.eval_components(x=x)
        print(out.fit_report(min_correl=0.5))
        f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace':0})                     
        ax1.set_ylabel("Voltage, V (V)", size=20)
        ax2.set_ylabel("Residual (V)")
        ax1.plot(x, init, 'k--', label="Initial guess")
        ax1.plot(x, y, 'bo', linestyle='-', markersize=2, label="Probe absorption depth")
        ax1.plot(x, out.best_fit, 'r-', label="Fit of the absorption depth")
        ax2.plot(x, out.residual, 'bo', markersize=2)
        ax1.legend(loc=3)
        plt.xlabel(r"Laser power, P ($\mu$A)", size=26)
        plt.tight_layout()
        plt.show()
