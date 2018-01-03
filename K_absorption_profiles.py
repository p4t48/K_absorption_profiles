import numpy as np
import matplotlib.pyplot as plt
import glob

dataFiles = glob.glob("20171222/*_5_*")
print(dataFiles)

f = open("%s" % dataFiles[0], "r")
f2 = open("%s" % dataFiles[1], "r")
data = np.fromfile(f, dtype=np.int16)
data2 = np.fromfile(f2, dtype=np.int16)
spectrum = data[1::4]
spectrum2 = data2[0::4]
trigger = data[3::4]
trigger2 = data2[3::4]
xVals = np.linspace(0,1,len(spectrum))
xVals2 = np.linspace(0,1,len(spectrum2))

print(data)

plt.plot(xVals, spectrum)
plt.plot(xVals, trigger)
plt.plot(xVals2, spectrum2)
plt.plot(xVals2, trigger2)
plt.show()
