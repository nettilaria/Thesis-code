#In this file I want to reproduce the  QY for XENONnT
# Qy=ne/E_0
'''Since I want to follow the approach of 
https://iopscience.iop.org/article/10.1088/1475-7516/2025/05/012/pdf (Valentina 2024)
Let me take the values of the parameters from here: https://arxiv.org/pdf/2406.13638 (Inference XENONnT (2025)).
In this paper they cite this one https://ieeexplore.ieee.org/document/7317823 from which I have taken the main formula
to implement the charge yield'''

#from xenonnt_plot_style import XENONPlotStyle as xps
#xps.use("xenonnt")
from scipy.interpolate import interp1d
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy import integrate
import inference_interface
from math import factorial
from multihist import poisson_1s_interval
from scipy.stats import binom
from scipy.special import gamma, factorial
from scipy import signal
import seaborn as sns
colors=sns.color_palette("rocket_r",n_colors=6)
colors_1=sns.color_palette("coolwarm")

#---------------- FIRST PART: ----------------------
# AIM: 1) Reproduction of the QY
#      2) Comparison with the collaboration one.  

Parameters={
    "alpha":0.92,
    "beta":334,
    "gamma":0.016,
    "delta":0.062,
    "zeta": 0.047,
    "W":0.0137,#kev
    "k":0.138,
    "f": 0.059
}


Z=54 #For Xe

Energy = np.linspace(0.1, 100, 500)


i=0


def z_raggomitolata(Electric_field):
    return Parameters["gamma"] * Electric_field**(-Parameters["delta"])
def epsilon(E):
    #The energy has to be in keV
    return 11.5*(E)*Z**(-7/3)
def L(E):
    g=3*epsilon(E)**(0.15)+0.7*epsilon(E)**(0.6)+epsilon(E)
    return Parameters["k"]*g/(1+Parameters["k"]*g)

def Ni(E,Electric_field):
    Ne_Ni=Parameters["alpha"]*Electric_field**(-Parameters["zeta"])*(1-np.exp(-Parameters["beta"]*epsilon(E)))
    return (E*L(E)/Parameters["W"])*1/(1+Ne_Ni)
def r(E,Electric_field):
    return 1-np.log(1+Ni(E,Electric_field)*z_raggomitolata(Electric_field))/(Ni(E,Electric_field)*z_raggomitolata(Electric_field))

def ne(E,Electric_field):
    Ne_Ni=Parameters["alpha"]*Electric_field**(-Parameters["zeta"])*(1-np.exp(-Parameters["beta"]*epsilon(E)))
    #r=1-(math.log(1+Ni*Parameters["Zeta"]))/(Ni*Parameters["Zeta"])
    return L(E)*(E/Parameters["W"])*(1/(1+Ne_Ni))*(1-r(E,Electric_field))
def QY(E,Electric_field):
    return ne(E,Electric_field)/E




CY_from_XENONnT= pd.read_csv('CY.txt', sep=";", header=None)

def figure_for_the_comparison():
    #Here I'm reproducing the fig. 4 of this https://ieeexplore.ieee.org/document/7317823 
    # I've chosen some electric field given by the paper, plum mine 23 V/cm. 
    #If you call this function, You can see how the charge yield change depending on the value of the field 
    fig, ax1 = plt.subplots(figsize=(3.3, 3))
    ax1.plot(CY_from_XENONnT[0],CY_from_XENONnT[1],label="XENONnT",color=colors_1[0])
    ax1.plot(Energy,QY(Energy,23),label="Qy reproduced",color=colors[0])
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Recoil Energy [keV]",fontsize=10)
    ax1.set_ylabel(r"Charge yield [$e^{-}$/keV]",fontsize=10)
    ax1.set_ylim(1.5,10)
    ax1.set_xlim(1,100)
    ax1.legend(ncol=2)
    ax1.tick_params(top=True, right=True, which='both')
    plt.show()

def figure_for_different_electric_field():
    #Here I want to compare my reproduction of the charge yield, to the one given by the collaboration (Digitalized by me)
    #If you call this function, you can see the comparison between the two plots. 
    fig, ax = plt.subplots(figsize=(3.3, 3))
    Electric_field_=[23,100,1000,2000,4000]
    labels=["23 V/cm","100 V/cm","1000 V/cm","2000 V/cm","4000 V/cm"]
    i=0
    for Electric_field in Electric_field_:
        ax.plot(Energy,QY(Energy,Electric_field),label=labels[i],color=colors[i])
        i+=1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Recoil Energy [keV]",fontsize=10)
    ax.set_ylabel(r"Charge yield [$e^{-}$/keV]",fontsize=10)
    ax.set_ylim(1.5,10)
    ax.set_xlim(1,100)
    ax.tick_params(top=True, right=True, which='both')
    ax.legend(loc="lower left", )
    plt.show()
figure_for_different_electric_field()

#------------------- END OF THE FIRST PART -------------------------------------------

# ----------------- SECOND PART .........................
# GOALS: 1) Find the value of T corresponding to cS2 binning
#        2) Find the number of CEVNS event for each bin 

# Something that you can find useful: How to find roots in python https://www.youtube.com/watch?v=768JiuFlBhc



g2_1= 16.5#First run
g2=16.9 #Second run

def S2_from_Recoil_Energy(E,gain):
    return E*QY(E,23)*gain


bin_1 = [ float("{0:.2f}".format(90)),float("{0:.2f}".format(120)),float("{0:.2f}".format(170.161773681641)),float("{0:.2f}".format(249.889739990234)),float("{0:.2f}".format(500.000000000000))]
print(bin_1)
bin_2 = [ float("{0:.2f}".format(90.000000000000)),float("{0:.2f}".format(120)),float("{0:.2f}".format(155.937850952148)),float("{0:.2f}".format(232.292022705078)),float("{0:.2f}".format(500.000000000000))]
print(bin_2)
guess_0= [0.70,0.98,1.40,2.25,5.19]
guess_1= [0.70,0.964,1.40,2.12,5.07]
i=0

print("----------- SR0 -------------")
for value_of_s2 in bin_1:
    def My_function(E):
        return value_of_s2-S2_from_Recoil_Energy(E,g2_1)
    def Recoil_Energy_from_S2(guessed_value):
        root=fsolve(My_function,guessed_value)
        return root
    
    print(' For S2: ',value_of_s2,' this is the value of T ',Recoil_Energy_from_S2(guess_0[i]))
    print('The number of electrons are:', ne(Recoil_Energy_from_S2(guess_0[i]),23))
    i+=1
i=0
print("----------- SR1 -------------")
for value_of_s2 in bin_2:
    def My_function(E):
        return value_of_s2-S2_from_Recoil_Energy(E,g2)
    def Recoil_Energy_from_S2(guessed_value):
        root=fsolve(My_function,guessed_value)
        return root
    
    print('For S2: ',value_of_s2,' this is the value of T ',Recoil_Energy_from_S2(guess_1[i]))
    print('The number of electrons are:', ne(Recoil_Energy_from_S2(guess_1[i]),23))
    i+=1





#Let me do the computation in KeV 
u_to_KeV=0.931494*1e6 #KeV

#constants
G_F = 1.1663787e-5*1e-12 #KeV^-2


Z=54    
A=131.29 #Avergaed atomic mass 
N=131-54
#binding_energy=(15.36*A-16.32*((A)**(2/3))-(90.46/A)*((A/2-Z)**2)-(0.6929*(Z**2))/(A**(1/3)))*1e3
m_N= 131.29*u_to_KeV#KeV

R_A=1.23* A**(1/3) #fm
sin2_theta_W=0.23857
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

y = (flux_8_B[1] * 5.46*1e6 )/1e3 # cm^-2 s^-1*KeV^-1
dflux= interp1d(x,y,bounds_error=False,fill_value=0.0) 




 #Combined efficiency for Sr0 and Sr1
def efficiency(T,text_file):
    df = pd.read_csv(text_file,sep=";",header=None)
    e= interp1d(df[0],df[1],bounds_error=False,fill_value=0.0)
    return e(T)


#---------------- Let me try to introduce the response of the detector -----------------
# I'm following: https://arxiv.org/abs/1606.09243
def total_number_of_available_quanta(T):
    T=T*1e3 #I have to put the recoil energy in eV
    return T/13.7




def S2(N_el_tilda):
    center=20*N_el_tilda
    std=7*np.sqrt(N_el_tilda)
    return 1/(std*np.sqrt(2*np.pi))*np.exp(-()**2/(2*(std**2)))


#---------------------------------------------------------------------------------------------
def poisson(T,n_i,n_f):
    i=n_i
    final_p=0
    while i<=n_f:
        final_p+=(ne(T,23)**i*np.exp(-ne(T,23)))/factorial(i)
        i+=1
    return final_p


#I have to define the other function for the computatio of the number of cevns event
def number_of_CEVNS_events_1(n_min,n_max):
    f = lambda E_nu,T: dflux(E_nu)*dsigma_VS_RecoilEnergy(E_nu,T)*efficiency(T,"Combined_efficiency.txt")*poisson(T,n_min,n_max)

    return integrate.dblquad(f,0.5,5.30,lambda T:1/2*(T + np.sqrt(T**2+2*m_N*T)),16.56e3 )
    
def number_of_CEVNS_evnts_0(T_min,T_max):
    # Minimum neutrino energy
    '''Double integration https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html'''

    f = lambda E_nu,T: dflux(E_nu)*dsigma_VS_RecoilEnergy(E_nu,T)*efficiency(T,"Combined_efficiency.txt")

    return integrate.dblquad(f,T_min,T_max,lambda T:1/2*(T + np.sqrt(T**2+2*m_N*T)),16.56e3 )

exposure=3.51 #SR0,SR1
run_1={
    "time":108.0/365, #days
    "FV":3.97 #tons
}
run_2={
    "time":208.5/365, #days
    "FV":4.10 #tons
}
mean_number_of_electron_1=[5,7,10,15,30]
mean_number_of_electron_2=[5,7,9,14,30]

"""print("------------ FIRST RUN FIRST ATTEMPT---------------")
print("1 bin: ",number_of_CEVNS_evnts_0(0.99136471, 1.4548847)[0]*365*24*60*60 *(6.022e29/131.29)*run_1["time"]*run_1["FV"])
print("2 bin: ",number_of_CEVNS_evnts_0(1.4548847, 2.26518045)[0]*365*24*60*60 *(6.022e29/131.29)*run_1["time"]*run_1["FV"])
print("3 bin: ",number_of_CEVNS_evnts_0(2.26518045,5.2201603)[0]*365*24*60*60 *(6.022e29/131.29)*run_1["time"]*run_1["FV"])
print("------------ FIRST RUN SECOND ATTEMPT---------------")
print("1 bin: ",number_of_CEVNS_events_1(7,10)[0]*365*24*60*60 *(6.022e29/131.29)*run_1["time"]*run_1["FV"])
print("2 bin: ",number_of_CEVNS_events_1(11,15)[0]*365*24*60*60 *(6.022e29/131.29)*run_1["time"]*run_1["FV"])
print("3 bin: ",number_of_CEVNS_events_1(16,30)[0]*365*24*60*60 *(6.022e29/131.29)*run_1["time"]*run_1["FV"])
print("----------- SECOND RUN FIRST ATTEMPT-------------")
print("1 bin: ",number_of_CEVNS_evnts_0(0.96625863,1.28503111)[0]*365*24*60*60 *(6.022e29/131.29)*run_2["time"]*run_2["FV"])
print("2 bin: ",number_of_CEVNS_evnts_0(1.28503111,2.02187888)[0]*365*24*60*60 *(6.022e29/131.29)*run_2["time"]*run_2["FV"])
print("3 bin: ",number_of_CEVNS_evnts_0(2.02187888,5.07011364)[0]*365*24*60*60 *(6.022e29/131.29)*run_2["time"]*run_2["FV"])
print("----------- SECOND RUN SECOND ATTEMPT-------------")
print("1 bin: ",number_of_CEVNS_events_1(7,9)[0]*365*24*60*60 *(6.022e29/131.29)*run_2["time"]*run_2["FV"])
print("2 bin: ",number_of_CEVNS_events_1(9,14)[0]*365*24*60*60 *(6.022e29/131.29)*run_2["time"]*run_2["FV"])
print("3 bin: ",number_of_CEVNS_events_1(15,30)[0]*365*24*60*60 *(6.022e29/131.29)*run_2["time"]*run_2["FV"])"""
print("----------- COMBINED RUN FIRST ATTEMPT-------------")
print("1 bin: ",number_of_CEVNS_evnts_0(0.71441802,1.28503111)[0]*365*24*60*60 *(6.022e29/131.29)*exposure)
print("2 bin: ",number_of_CEVNS_evnts_0(1.28503111,2.02187888)[0]*365*24*60*60 *(6.022e29/131.29)*exposure)
print("3 bin: ",number_of_CEVNS_evnts_0(2.02187888,5.07011364)[0]*365*24*60*60 *(6.022e29/131.29)*exposure)


