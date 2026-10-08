from xenonnt_plot_style import XENONPlotStyle as xps
from scipy.interpolate import interp1d

xps.use("xenonnt")
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import inference_interface
from multihist import poisson_1s_interval
from scipy.optimize import fsolve
from scipy import integrate



livetime=3.249665
#---------------------- NEW PART -------------------------
u_to_KeV=0.931494*1e6 #KeV
G_F = 1.1663787e-5*1e-12 #KeV^-2
m_N= 131.29*u_to_KeV #KeV
Z=54    
A=131.29 #Avergaed atomic mass 
N=A-Z
R_A=1.23* A**(1/3) #fm
sin2_theta_W=0.23120
g_p=(1-4*sin2_theta_W)/2
g_n=-0.5
Q_V=g_p*Z+g_n*N

def Iorder_bessel(x):
    return np.sin(x)/x**2-np.cos(x)/x

    
def F_W(q):   
    fm_to_keV=1/0.197 * 1e-6 #KeV^-1
    return (3*Iorder_bessel(q*R_A*fm_to_keV)/(q*R_A*fm_to_keV))*(1/(1+(q*0.7*fm_to_keV)**2))
   
                                                             
def dsigma_VS_RecoilEnergy(E_nu,T_N):
    q = np.sqrt(2*m_N*T_N)
    dsigma_1=(((G_F**2)*m_N)/(np.pi))*(Q_V**2)*(F_W(q)**2)*(1-(m_N*T_N)/(2*E_nu**2))# this one is in KeV^-3
    dsigma_1=dsigma_1* (1.973269804e-8)**2 #this result is cm^2/keV (Important because of the mutiplication with the flux)
    return dsigma_1

flux_8_B= pd.read_csv('8B.txt', sep=r'\s+', header=None)

#Neutrino energy from MeV to KeV
x = flux_8_B[0]* 1e3

# Flux: per MeV -> per keV
#Total flux: 5.46 e6
y = (flux_8_B[1] * 5.46*1e6 )/1e3 # cm^-2 s^-1*KeV^-1
dflux= interp1d(x,y,bounds_error=False,fill_value=0.0) 



#plt.plot(df[0],df[1])

def efficiency(T,observable,bin):
    templates_cevns = inference_interface.template_to_multihist(
        f"templates/sr2/mono/template_XENONnT_sr2_mono_{T:.1f}00_cevns_tly_0.0_tqy_0.0.h5", hist_name="template")
    return templates_cevns.project(observable).histogram[bin]



#print("This is the efficiency of 0.500 keV ",efficiency(0.500,"cs2",1))






def number_of_CEVNS_evnts(observable,bin):

    # Minimum neutrino energy
    '''Double integration https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html'''

    f = lambda E_nu,T: dflux(E_nu)*dsigma_VS_RecoilEnergy(E_nu,T)*efficiency(T,observable,bin)
    integral=integrate.dblquad(f,0.5,5,lambda T:np.sqrt(m_N*T/2),16.56e3)
    integral=integral[0]*365*24*60*60 *(6.022e29/131.29)*livetime
    return integral

cevns_events_1_cs2=0
"""
print("--------- CS2-----------")
print(number_of_CEVNS_evnts("cs2",0))
print(number_of_CEVNS_evnts("cs2",1))
print(number_of_CEVNS_evnts("cs2",2))
print("--------- s2_shadow_s2_time_shadow -----------")
print(number_of_CEVNS_evnts("s2_shadow_s2_time_shadow",0))
print(number_of_CEVNS_evnts("s2_shadow_s2_time_shadow",1))
print(number_of_CEVNS_evnts("s2_shadow_s2_time_shadow",2))
print("--------- s1_bdt_score -----------")
print(number_of_CEVNS_evnts("s1_bdt_score",0))
print(number_of_CEVNS_evnts("s1_bdt_score",1))
print(number_of_CEVNS_evnts("s1_bdt_score",2))
print("--------- s2_bdt_score -----------")
print(number_of_CEVNS_evnts("s2_bdt_score",0))
print(number_of_CEVNS_evnts("s2_bdt_score",1))
print(number_of_CEVNS_evnts("s2_bdt_score",2))"""


    

data = pd.read_csv(f"data/sr2.csv", index_col=0) 



with open(f"binning/sr2.yaml", "r") as f:binning = yaml.safe_load(f)

'''print(binning) gives me the following result: 
{'cs2': [90.0, 170.161773681641, 249.889739990234, 500.0],
's2_shadow_s2_time_shadow_quantile': [0.0, 0.333333333333, 0.666666666667, 1.0],
's1_bdt_score': [0.0, 0.282301220403, 0.498535093, 1.0],
's2_bdt_score': [0.8069558, 0.883306158412, 0.923731456497, 1.0]}

So it is a dictionary 

'''


'''Now I want to read the three backgrounds (AC, Neutron and ER)'''

templates_AC = inference_interface.template_to_multihist(
        f"templates/sr2/ac/template_XENONnT_sr2_ac_cevns.h5", hist_name="template")
templates_ER = inference_interface.template_to_multihist(
        f"templates/sr2/er/template_XENONnT_sr2_er_cevns.h5", hist_name="template")
#-------------------------------------------------------
# From https://github.com/XENONnT/cevns_data_release/blob/master/README.md:
# tly_* and tqy_* — correspond to the nuisance parameters modeling the light yield and charge yield uncertainties
# From https://arxiv.org/pdf/2604.06002:
# The corresponding uncertainties are modeled by nuisance parameters, tLy and tQy, defined such that tLy = 0 (tQy = 0)
# corresponds to the median of Ly (Qy), while tLy =±1 (tQy = ±1) corresponds to the ±1σ quantiles.
#-------------------------------------------------------

templates_Neutron=inference_interface.template_to_multihist(
            f"templates/sr2/rg/template_XENONnT_sr2_rg_cevns_tly_0.0_tqy_0.0.h5", hist_name="template")


axis_labels = {
    "cs2": "cS2", #I've removed the "quantile" (because of the observable that we want to consider)
    #The c in front of S2 is because it is the corrected signal S2 (https://arxiv.org/pdf/2412.10451)
    "s2_shadow_s2_time_shadow_quantile": "Quantile of $\mathrm{S2_{pre}}$ / $\Delta t_\mathrm{pre}$",
    "s1_bdt_score": "Quantile of S1 BDT score",
    "s2_bdt_score": "Quantile of S2 BDT score",
}

#This is just to make subplots: in this case you have 4 rows all in one column 
fig, axes = xps.subplots(nrows=4, ncols=1, rescale=(1.0, 2.0)) 
Events_AC=0
h_stacked_ER=0
for i, label in enumerate(axis_labels.keys()):
    h_data = 0
    h_templates_AC = 0
    h_templates_ER=0
    h_templates_Neutron=0
 

    h_data += np.histogram(data[label], bins=binning[label])[0]
    h_templates_AC += templates_AC.project(axis=i).histogram * livetime
    #print("AC in ",label,": ",h_templates_AC,"\n")
    
    
    h_templates_ER += templates_ER.project(axis=i).histogram * livetime
    #print("ER in ",label,": ",h_templates_ER,"\n")


    h_templates_Neutron+= templates_Neutron.project(axis=i).histogram * livetime
    #print("Neutron in ",label,": ",h_templates_Neutron,"\n")

    #Let me now build stack histograms.

    h_stacked_Neutron=h_templates_Neutron+h_templates_AC
    h_stacked_ER=h_stacked_Neutron+h_templates_ER

    if(label=='cs2'):
        bins=np.array(binning[label]) 
        '''We do no want the quantile of the cs2 observable, 
        so I'm using the original bins. I'm converting to an array because of:(bins[1:] + bins[:-1]) / 2
        '''
    else:
        bins = np.linspace(0, 1, 4)

    
    
    axes[i].fill_between(
        bins,
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
    if i==0:
            axes[i].fill_between(
                bins,
                np.hstack([0, h_stacked_ER]),
                np.hstack([0, h_stacked_ER+[3.111041111940574,3.1264454183089807,0.9326182204560974]]),
                step="pre",
                facecolor="#57BE84",
                edgecolor=None,
                label="ER"
            )
            axes[i].step(
                bins,
                np.hstack([0, [  8.20068386522851,  8.208371789169353, 6.563994217724898
]]),
                color="indigo",
        )

   
    elif i==1:
        axes[i].fill_between(
                        bins,
                        np.hstack([0, h_stacked_ER]),
                        np.hstack([0, h_stacked_ER+[3.496111163716388,2.2985992506302733,1.3753943363589898]]),
                        step="pre",
                        facecolor="#57BE84",
                        edgecolor=None,
                        label="ER"
                    )
        axes[i].step(
                bins,
                np.hstack([0, [ 8.960781480767995 ,7.57456816284595  , 6.56641302253901]]),
                color="indigo",
        )

   
    elif i==2:
        axes[i].fill_between(
                        bins,
                        np.hstack([0, h_stacked_ER]),
                        np.hstack([0, h_stacked_ER+[0.41236651016907877,1.4403832971734825,5.317354943363091]]),
                        step="pre",
                        facecolor="#57BE84",
                        edgecolor=None,
                        label="ER"
                    )
        axes[i].step(
                bins,
                np.hstack([0, [  5.48442212315583,  6.641084965107312, 11.012716217257577]]),
                color="indigo",
        )

    else:
        axes[i].fill_between(
                        bins,
                        np.hstack([0, h_stacked_ER]),
                        np.hstack([0, h_stacked_ER+[1.2601479231887323,2.4613631793533943,3.4485936481635244]]),
                        step="pre",
                        facecolor="#57BE84",
                        edgecolor=None,
                        label="ER"
                    )
        axes[i].step(
                bins,
                np.hstack([0, [  6.521370216855889, 7.780906469634168 ,8.919474980476963 ]]),
                color="indigo",
        )




   
        
    axes[i].set_xlabel(axis_labels[label])
    if label=='cs2':
        axes[i].set_xlim(100,500)
    else:
        axes[i].set_xlim(0,1)

    axes[i].set_ylim(bottom=0)



fig.supylabel("Events per bin")


axes[0].legend(bbox_to_anchor=[0.5, 1.5],loc='upper center',ncol=5)
plt.savefig('CEVNS_1.pdf')

plt.show()