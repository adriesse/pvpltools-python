"""
This script demonstrates reading IEC 61853-1 matrix measurements
stored in the DataPlusMeta format.

Copyright (c) 2019-2020 Anton Driesse, PV Performance Labs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')

from pvpltools.dataplusmeta import DataPlusMeta

#%%

# obtain some matrix data
dpm = DataPlusMeta.from_txt('data/CS5P-220M.txt')
name = dpm.meta['name']
mtx = dpm.data

# calculate the relative efficiency compared to stc
stc = mtx.query('irradiance == 1000 and temperature == 25').mean()
mtx['eta_rel'] = mtx.p_mp / mtx.irradiance * stc.irradiance / stc.p_mp

# create a pivit table for easy plotting
eta_rel = mtx.pivot(index='irradiance', columns='temperature', values='eta_rel')
print(eta_rel)
print(eta_rel.T)

# plot vs irradiance
fig, ax = plt.subplots(1,1, num=name+' Irradiance')
ax.set_prop_cycle('color', plt.cm.rainbow(np.linspace(0,1,len(eta_rel.columns))))
eta_rel.plot(style='s-', lw=2, ax=ax)

# make nice
plt.xlim(0, 1250)
plt.ylim(0.68, 1.12)
plt.grid()
plt.legend(loc='lower right', title='Temperature')
plt.title(name)
plt.xlabel('Irradiance (W/m²)')
plt.ylabel('Efficiency relative to STC efficiency')

# plt vs temperature
fig, ax = plt.subplots(1,1, num=name+' Temperature')
ax.set_prop_cycle('color', plt.cm.rainbow(np.linspace(0,1,len(eta_rel.index))))
eta_rel.transpose().plot(style='s-', lw=2, ax=ax)

# add gamma
gammafun = lambda T: 1 + (T - 25) * dpm.meta['datasheet']['gamma_mp'] / 100
trange = np.array([0, 100])
ax.plot(trange, gammafun(trange), 'k--', lw=2, label='gamma Pmax')

# make nice
plt.xlim(12, 78)
plt.ylim(0.68, 1.12)
plt.grid()
plt.legend(loc='upper right', title='Irradiance')
plt.title(name)
plt.xlabel('Temperature (°C)')
plt.ylabel('Efficiency relative to STC efficiency')
