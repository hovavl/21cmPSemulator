import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
from distinctipy import distinctipy
path = '/Users/hovavlazare/Documents/university/21cm_Project/L_x_files/tmp'
vals = []
counter = 0
for file in os.listdir(path):
    filepath = f'{path}/{file}'
    with open(filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        Lx = []
        prob = []
        for row in csv_reader:
            Lx += [float(row[0])]
            prob += [float(row[1])]
        vals += [[Lx,prob]]

minx = 100
maxx = 0
for i,val in enumerate(vals):
    val = np.array(val)


    logic_lx = val[0] < 41
    vals[i][0] = val[0][logic_lx]
    vals[i][1] =  gaussian_filter1d(val[1][logic_lx],3)
    if np.min(vals[i][0]) < minx:
        minx = np.min(vals[i][0])
    if np.max(vals[i][0]) > maxx:
        maxx = np.max(vals[i][0])



#xmax = np.max(vals[:][0])


plt.rc('text', usetex=True) #render font for tex
plt.rc('font', family='TimesNewRoman', size=30) #use font
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = 15 # pad is in points...


labels = ['Emulator based MCMC with power spectrum','Emulator based MCMC without power spectrum',
          'HERA MCMC with power spectrum', 'HERA MCMC without power spectrum'
          ]
colors = ['red', 'blue', 'purple', 'black']

plt.figure(figsize=(20,14))

for i, cap in enumerate(vals):
    plt.plot(cap[0],cap[1], linewidth=10, label = labels[i], color = colors[i])
plt.legend(prop={"size":24})
plt.xlim(minx,maxx)
plt.xticks(fontsize=26)
plt.yticks(fontsize = 26)

plt.xlabel(r'$\log_{10}\frac{L_{\rm X<2 \, keV/SFR}}{\rm erg\, s^{-1}\,M_{\odot}^{-1}\,yr}$', fontsize = 34)
plt.ylabel(r'$\rm Probability$', fontsize = 34)
plt.savefig('tmp')
plt.show()
