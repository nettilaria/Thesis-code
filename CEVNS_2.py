from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import inference_interface
from multihist import poisson_1s_interval

#----------------------------------------------------------
#     In this file I'm computing the 4 plots for sr2
#     Picture to reproduce: Fig. 10 of https://arxiv.org/pdf/2603.00554
#----------------------------------------------------------
livetime=3.249665

data = pd.read_csv(f"data/sr2.csv", index_col=0)

with open(f"binning/sr2.yaml", "r") as f:binning = yaml.safe_load(f)

'''Now I want to read the three backgrounds (AC, Neutron and ER)'''

templates_AC = inference_interface.template_to_multihist(
        f"templates/sr2/ac/template_XENONnT_sr2_ac_cevns.h5", hist_name="template")
templates_ER = inference_interface.template_to_multihist(
        f"templates/sr2/er/template_XENONnT_sr2_er_cevns.h5", hist_name="template")

'''In this case i do not have an unique file in the folder.
If I want to read them all, I have to build a loop'''

tly_values = ["-3.0", "-2.0", "-1.0", "0.0", "1.0", "2.0", "3.0"]
tqy_values = ["-3.0", "-2.0", "-1.0", "0.0", "1.0", "2.0", "3.0"]
templates_Neutron=[ ]

for tly in tly_values:
    row=[]
    for tqy in tqy_values:
        template=0
        template=inference_interface.template_to_multihist(
            f"templates/sr2/rg/template_XENONnT_sr2_rg_cevns_tly_{tly}_tqy_{tqy}.h5", hist_name="template")
        row.append(template)
    templates_Neutron.append(row)
    
#Just a check that I did not build an empty object!!!
#for i in range(7):
#    for j in range(7):
#        print(templates_Neutron[i][j])

axis_labels = {
    "cs2": "Quantile of cS2",
    "s2_shadow_s2_time_shadow_quantile": "Quantile of $\mathrm{S2_{pre}}$ / $\Delta t_\mathrm{pre}$",
    "s1_bdt_score": "Quantile of S1 BDT score",
    "s2_bdt_score": "Quantile of S2 BDT score",
}
fig, axes = xps.subplots(nrows=4, ncols=1, rescale=(1.0, 2.0))
for i, label in enumerate(axis_labels.keys()):
    h_data = 0
    h_templates_AC = 0
    h_templates_ER=0
    h_templates_Neutron=0

    h_data += np.histogram(data[label], bins=binning[label])[0]
    h_templates_AC += templates_AC.project(axis=i).histogram * livetime
    
    h_templates_ER += templates_ER.project(axis=i).histogram * livetime
    for i_ in range(len(tly_values)):
        for j in range(len(tqy_values)):
            h_templates_Neutron= templates_Neutron[i_][j].project(axis=i).histogram * livetime
    #Let me now build stack histograms.

    h_stacked_Neutron=h_templates_Neutron+h_templates_AC
    h_stacked_ER=h_stacked_Neutron+h_templates_ER

    bins = np.linspace(0, 1, 4)

    axes[i].fill_between(
        bins,
        0,
        np.hstack([0, h_templates_AC]),
        step="pre",
        facecolor="#9A4C78",
        edgecolor=None,
    )
    axes[i].fill_between(
            bins,
            np.hstack([0, h_templates_AC]),
            np.hstack([0, h_stacked_Neutron]),
            step="pre",
            facecolor="#FFCE73",
            edgecolor=None,
        )
    axes[i].fill_between(
                bins,
                np.hstack([0, h_stacked_Neutron]),
                np.hstack([0, h_stacked_ER]),
                step="pre",
                facecolor="#6787BC",
                edgecolor=None,
            )
    ylow, yhigh = poisson_1s_interval(h_data, fc=True)
    axes[i].errorbar(
        (bins[1:] + bins[:-1]) / 2,
        y=h_data,
        yerr=[h_data - ylow, yhigh - h_data],
        color="k",
        fmt="o",
    )
    axes[i].set_xlabel(axis_labels[label])
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(bottom=0)

    
    
        
   

fig.supylabel("Events per bin")

plt.show()