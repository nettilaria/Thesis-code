from xenonnt_plot_style import XENONPlotStyle as xps

xps.use("xenonnt")
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import inference_interface
from multihist import poisson_1s_interval

#----------------------------------------------------------
#     In this file I'm computing the 4 plots for sr0
#     Picture to reproduce: Fig. 10 of https://arxiv.org/pdf/2603.00554
#----------------------------------------------------------
livetime=1.174398

data = pd.read_csv(f"data/sr0.csv", index_col=0)

with open(f"binning/sr0.yaml", "r") as f:binning = yaml.safe_load(f)

'''Now I want to read the three backgrounds (AC, Neutron and ER)'''

templates_AC = inference_interface.template_to_multihist(
        f"templates/sr0/ac/template_XENONnT_sr0_ac_cevns.h5", hist_name="template")
templates_ER = inference_interface.template_to_multihist(
        f"templates/sr0/er/template_XENONnT_sr0_er_cevns.h5", hist_name="template")
#-------------------------------------------------------
# From https://github.com/XENONnT/cevns_data_release/blob/master/README.md:
# tly_* and tqy_* — correspond to the nuisance parameters modeling the light yield and charge yield uncertainties
# From https://arxiv.org/pdf/2604.06002:
# The corresponding uncertainties are modeled by nuisance parameters, tLy and tQy, defined such that tLy = 0 (tQy = 0)
# corresponds to the median of Ly (Qy), while tLy =±1 (tQy = ±1) corresponds to the ±1σ quantiles.
#-------------------------------------------------------

templates_Neutron=inference_interface.template_to_multihist(
            f"templates/sr0/rg/template_XENONnT_sr0_rg_cevns_tly_0.0_tqy_0.0.h5", hist_name="template")


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

    h_templates_Neutron+= templates_Neutron.project(axis=i).histogram * livetime
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
        label="AC"
    )
    axes[i].fill_between(
            bins,
            np.hstack([0, h_templates_AC]),
            np.hstack([0, h_stacked_Neutron]),
            step="pre",
            facecolor="#FFCE73",
            edgecolor=None,
            label="Neutron"
        )
    axes[i].fill_between(
                bins,
                np.hstack([0, h_stacked_Neutron]),
                np.hstack([0, h_stacked_ER]),
                step="pre",
                facecolor="#6787BC",
                edgecolor=None,
                label="ER"
            )
    ylow, yhigh = poisson_1s_interval(h_data, fc=True)
    axes[i].errorbar(
        (bins[1:] + bins[:-1]) / 2,
        y=h_data,
        yerr=[h_data - ylow, yhigh - h_data],
        color="k",
        fmt="o",
        label="Data"
    )
   
        
    axes[i].set_xlabel(axis_labels[label])
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(bottom=0)



fig.supylabel("Events per bin")



plt.show()