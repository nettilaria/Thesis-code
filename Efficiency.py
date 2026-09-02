from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")
#import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import inference_interface
from multihist import poisson_1s_interval

'''I've taken this part from https://github.com/XENONnT/cevns_data_release/blob/master/plot.ipynb.
I've only modified the final result in order to obtain the acceptance'''

mono_energies = np.linspace(0.5, 5.0, 46)
efficiency = dict()
for sr in ["sr0", "sr1", "sr2"]:
    efficiency[sr] = list()
    for T in mono_energies:
        efficiency[sr].append(
            inference_interface.template_to_multihist(
                f"templates/{sr}/mono/template_XENONnT_{sr}_mono_{T:.3f}_cevns_tly_0.0_tqy_0.0.h5",
                hist_name="template",
            ).n
        )
fig, ax = plt.subplots(1, 1, figsize=(4, 3))

'''-----------------------------------------------
    Picture with the Esposure (trying to reproduce the picture of https://arxiv.org/pdf/2604.06002)
    Data: 1 run: (3.97 t)*(108.0 days)
          2 run: (4.10 t)*(208.5 days)
          3 run: (4.14 t)*(286.5 days)
          total: (12,21 t)*(603 days)

FROM NOW ON: "EFFICIENCY" MEANS "ACCEPTANCE"!!
------------------------------------------------'''


runs_tons=[3.97,4.10,4.14]

runs_years= [108.0/365,208.5/365,286.5/365]

i=0
efficiency_total=0
for sr in ["sr0", "sr1", "sr2"]:
    efficiency[sr]=np.array(efficiency[sr], dtype=float)*runs_years[i]*runs_tons[i]
    efficiency_total+= efficiency[sr]
    ax.plot(mono_energies,efficiency[sr], label=sr.upper())
    print("Exposure:" ,runs_years[i]*runs_tons[i],"\n")

    i=i+1
efficiency_total=efficiency_total/(12.21*603/365)
ax.plot(mono_energies, efficiency_total, label="Combined Acceptance",c="#FFA952")

ax.legend(ncols=3)
ax.set_xlim(mono_energies[0], mono_energies[-1])
ax.set_xlabel("NR energy [keV]")
ax.set_ylabel("Acceptance")
#plt.savefig('acceptance.pdf')
plt.show()