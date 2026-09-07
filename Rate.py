from xenonnt_plot_style import XENONPlotStyle as xps
xps.use("xenonnt")
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import inference_interface
from multihist import poisson_1s_interval

"The theoretical equations are taken from https://arxiv.org/pdf/2603.00554"

#Let me do the computation in KeV 
u_to_KeV=0.931494*1e6 #KeV

#constants
G_F = 1.1663787e-5*1e-12 #KeV^-2
m_N= 131.29*u_to_KeV #KeV
Z=54
A=131.29 #Avergaed atomic mass 
N=A-Z
R_A=1.23* A**(1/3) #fm
sin2_theta_W=0.23120
g_p=(0.5-2*sin2_theta_W)/2
g_n=-0.5
Q_V=g_p*Z+g_n*N


def Iorder_bessel(x):
    return np.sin(x)/x**2-np.cos(x)/x

    
def F_W(q):   
    fm_to_keV=1/0.197 * 1e-6 #KeV^-1
    return (3*Iorder_bessel(q*R_A*fm_to_keV)/(q*R_A*fm_to_keV))*(1/(1+(q*0.7*fm_to_keV)**2))
   
                                                             
def dsigma_VS_RecoilEnergy(T_N,E_nu):
    q = np.sqrt(2*m_N*T_N)
    dsigma_1=(((G_F**2)*m_N)/np.pi)*(Q_V**2)*(F_W(q)**2)*(1-(m_N*T_N)/(2*E_nu**2))# this one is in KeV^-3
    dsigma_1=dsigma_1* (1.973269804e-8)**2 #this result is cm^2/keV (Important because of the mutiplication with the flux)
    return dsigma_1 

flux_8_B= pd.read_csv('8B.txt', sep=r'\s+', header=None)

#Neutrino energy from MeV to KeV
x = flux_8_B[0]* 1e3

# Flux: per MeV -> per keV
#Total flux: 5 e6
y = (flux_8_B[1] * 5e-04*1e10 )/1e3 # cm^-2 s^-1*KeV^-1

mono_energies = np.linspace(0.5, 5.0, 46) #I've used the same of the efficiency. 

'''From https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html:
 x and y are arrays of values used to approximate some function f:
 y = f(x). This class returns a function whose call method uses interpolation to find the value of new points.'''

dflux = interp1d(x,y)


def dR_VS_RecoilEnergy(T_N, dflux):
    # How to make integrals in python https://docs.scipy.org/doc/scipy-1.18.0/tutorial/integrate.html

    # Minimum neutrino energy
    E_min = np.sqrt(m_N*T_N/2)
  

    # Maximum neutrino energy
    E_max = 16e3  # KeV

    if E_min >= E_max:
        return 0.0

    def integrand(E_nu):
        
        return (

            dflux(E_nu)
            * dsigma_VS_RecoilEnergy(T_N, E_nu)
        )
    '''From: https://docs.scipy.org/doc/scipy/tutorial/integrate.html:
    The return value is a tuple, with the first element holding the estimated value of the integral
     and the second element holding an estimate of the absolute integration error'''
    
    integral= integrate.quad(
        integrand,
        E_min,
        E_max
    )[0] 
 
    #print(integral)

    return integral



rate_values = []

for T in mono_energies:
    rate = dR_VS_RecoilEnergy(T, dflux)
    rate_values.append(rate)
#Units: s^-1 KeV^-1
rate_values = np.array(rate_values)
#Units: ton ^-1 s^-1 KeV^-1
rate_values=rate_values*365*24*60*60 *(6.022e29/131.29) 


# XENONnT efficiency from https://zenodo.org/records/20576156



efficiency = {}

for sr in ["sr0", "sr1", "sr2"]:

    efficiency[sr] = []

    for T in mono_energies:
        
        template = inference_interface.template_to_multihist(
            f"templates/{sr}/mono/"
            f"template_XENONnT_{sr}_mono_{T:.3f}_cevns_tly_0.0_tqy_0.0.h5",
            hist_name="template",
        )

        efficiency[sr].append(template.n)

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plt.plot(mono_energies, rate_values, label="w/o acceptance",c='#B22222' ,linewidth=1.3,linestyle='--')

#plt.plot(mono_energies, efficiency["sr0"]*rate_values, label="SR0",linewidth=1.1)
#plt.plot(mono_energies, efficiency["sr1"]*rate_values, label="SR0",linewidth=1.1)
#plt.plot(mono_energies, efficiency["sr2"]*rate_values, label="SR0",linewidth=1.1)

'''-----------------------------------------------
    Picture with the Esposure (trying to reproduce the picture of https://arxiv.org/pdf/2604.06002)
    Data: 1 run: (3.97 t)*(108.0 days)
          2 run: (4.10 t)*(208.5 days)
          3 tun: (4.14 t)*(286.5 days)
          FROM NOW ON: "EFFICIENCY" MEANS "ACCEPTANCE"!!
------------------------------------------------'''

runs_tons=[3.97,4.10,4.14]

runs_years= [108.0/365,208.5/365,286.5/365]
i=0
efficiency_total=0
for sr in ["sr0", "sr1", "sr2"]:
    efficiency[sr]=np.array(efficiency[sr], dtype=float)*runs_years[i]*runs_tons[i]
    efficiency_total+= efficiency[sr]
    print("Exposure ",i,": ", runs_years[i]*runs_tons[i],"\n")

    i=i+1

efficiency_total=efficiency_total/(3.97* 108.0/365+4.10*208.5/365+4.14*286.5/365)
print("Total exposure:",3.97* 108.0/365+4.10*208.5/365+4.14*286.5/365)
rate_w_acceptance = rate_values * efficiency_total
plt.plot(mono_energies, rate_w_acceptance, label="w/ acceptance",linewidth=1.1,c="#FFA952")



plt.xlabel(r"$T_N$ [keV]", fontsize=10)
plt.ylabel(
    r"$dR/dT_N$ "
    r"[t$^{-1}$ yr$^{-1}$ keV$^{-1}$]",
    fontsize=10
)

plt.yscale("log")
plt.xlim(0.5, 4.5)
plt.ylim(0.1, 1e3)
plt.legend()
#plt.savefig('Rate_acceptance.pdf')
plt.show()



