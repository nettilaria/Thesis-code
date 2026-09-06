'''Files to be read for the neutrino flux. 
I've obtained them from https://www.sns.ias.edu/~jnb/'''

#For the data analysis Pandas is used.  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

files=["8B.txt","13N.txt","15O.txt","17F.txt","hep.txt","pp.txt"]

'''--------Tab1 of the thesis-----------'''
total_fluxes = np.array([
    0.582e-03,
    0.571e-01,
    0.503e-01,
    0.591e-03,
    0.788e-06,
    0.594e+01,
    0.486e+00*0.90,#Be7
    0.486e+00*0.10,#Be7 I've multiplied the solar fluxes by the Br of the reaction. 
    0.140e-01
])
total_fluxes = total_fluxes * 1e10
labels = ['$^{8}$B', '$^{13}$N', '$^{15}$O', '$^{17}$F', 'hep', 'pp', '$^{7}$Be', '$^{7}$Be*','pep']
colors=['#1F77B4','#D62728','#9467BD','#8C564B', '#FF7F0E','#2CA02C']



for i in range(9):
    x=0
    y=0
#------ Continuum lines -------------------
    if i<6:
        data=pd.read_csv(filepath_or_buffer=files[i], #Type of file
                        sep='\s+', #Not all the files are saparated by Tab
                        header=None, #No header in my case
                        ) 
        x=data[0]
        y=data[1]*total_fluxes[i]
        if i==0 or i==4:
            plt.plot(x, y, label=labels[i],c=colors[i],linewidth=1.5)
            if i==0:
                plt.fill_between(x,y, color=colors[i], alpha=0.1)
                plt.text(x[300],0.3e7,labels[i], c=colors[i], fontsize=12)
            else:
                plt.text(x[300],1e2,labels[i], c=colors[i], fontsize=12)
        elif i==1:
            plt.text(x[20],y[40],labels[i], c=colors[i], fontsize=12)
            plt.plot(x, y, label=labels[i],c=colors[i],linewidth=1.5,linestyle='--')
        
        else:
            if i== 5:#pp
                plt.plot(x, y, label=labels[i],c=colors[i],linewidth=1.5)
                plt.text(x[16],y[7],labels[i], c=colors[i], fontsize=12)
        
            else:
                plt.plot(x, y, label=labels[i],c=colors[i],linewidth=1.5,linestyle='--')
                plt.text(x[18],y[6],labels[i], c=colors[i], fontsize=12)



    #-------- Vertical lines --------------------
    else:
        if i==6: #Be7 
            plt.plot([0.8613, 0.8613], [1e-5, total_fluxes[i]],label=labels[i],c='#000000',linewidth=1.5)
            plt.text(0.55,160,labels[i],c='#000000', fontsize=12)    
        elif i==7: #Be7
            plt.plot([0.3833, 0.3833], [1e-5, total_fluxes[i]], label=labels[i],c='#000000',linewidth=1.5)
            plt.text(0.20,160,labels[i], c='#000000', fontsize=12)
               
        elif i==8:#pep
            plt.plot([1.44, 1.44], [1e-5, total_fluxes[i]], label=labels[i],c='#000000',linewidth=1.5)
            plt.text(1.2,0.8e9,labels[i], c='#000000', fontsize=12)

    
    

plt.xlabel('Neutrino energy [MeV]', fontsize=14)
plt.ylabel('Flux [$cm^{-2} s^{-1} (MeV^{-1})$]', fontsize=14)
plt.xlim(0.05, 25) 
plt.ylim(1e-2, 1e12)  
plt.yscale('log')
plt.xscale('log')
plt.title('Solar neutrinos fluxes',fontsize=15)

plt.show()