import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''Files to be read for the neutrino flux. 
I've obtained them from https://www.sns.ias.edu/~jnb/'''

files = {
    '8B': '8B.txt',
    '13N': '13N.txt',
    '15O': '15O.txt',
    '17F': '17F.txt',
    'hep': 'hep.txt',
    'pp': 'pp.txt',
#The following three are here only for loop convenience, I'm not reading any file.
    '7Be': 'Be7(861keV).txt', 
    '7Be1': 'Be7(384keV).txt',
    'pep': 'pep.txt',
}

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

'''----Appearence of the picture------'''

labels = ['$^{8}$B', '$^{13}$N', '$^{15}$O', '$^{17}$F', 'hep', 'pp', '$^{7}$Be', '$^{7}$Be*','pep']
color = {
    'pp':   '#2CA02C',   
    '13N':  '#D62728',   
    '15O':  '#9467BD',   
    '17F':  '#8C564B',   
    '8B':   '#1F77B4',   
    'hep':  '#FF7F0E',   
    '7Be':  '#000000',   
    '7Be1': '#000000',   
    'pep':  '#000000',   
}
i=0

"---------The cycle is so long just because each channel has its own color and lable ------"
for type in files:
    data=files[type]
    df = pd.read_csv(data,  sep=r'\s+',header=None)
    #************* Lines *****************

    if i==6: #Be7 
        x=0.8613
        y=total_fluxes[i]
        plt.plot([x, x], [1e-5, y],label=labels[i],c=color[type],linewidth=1.5)
        plt.text(0.69,160,labels[i], c=color[type],va='center', ha='center', fontsize=12)
        
    elif i==7: #Be7
        x=0.3833
        y=total_fluxes[i]
        plt.plot([x, x], [1e-5, y], label=labels[i],c=color[type],linewidth=1.5)
        plt.text(0.29,160,labels[i], c=color[type],va='center', ha='center', fontsize=12)
       
    elif i==8:#pep
        x=1.44
        y=total_fluxes[i]
        plt.plot([x, x], [1e-5, y], label=labels[i],c=color[type],linewidth=1.5)
        plt.text(1.45,3e9,labels[i], c=color[type],va='center', ha='center', fontsize=12)

    #*************************************** 
    #************* Continuous spectra *****************
    else:
        x=df[0]
        y=df[1]*total_fluxes[i]
        
        if i==0 or i==4:
            
            plt.plot(x, y, label=labels[i],c=color[type],linewidth=1.5)
            if i==0:
                plt.fill_between(x,y, color=color[type], alpha=0.1)
                plt.text(x[300],0.3e7,labels[i], c=color[type],va='center', ha='center', fontsize=12)
            else:
                plt.text(x[300],1e2,labels[i], c=color[type],va='center', ha='center', fontsize=12)
        elif i==1:
            plt.text(x[20],y[40],labels[i], c=color[type],va='center', ha='center', fontsize=12)
            plt.plot(x, y, label=labels[i],c=color[type],linewidth=1.5,linestyle='--')

        else:
            if i== 5:#pp
                plt.plot(x, y, label=labels[i],c=color[type],linewidth=1.5)
                plt.text(x[16],y[9],labels[i], c=color[type],va='center', ha='center', fontsize=12)

            else:
                plt.plot(x, y, label=labels[i],c=color[type],linewidth=1.5,linestyle='--')
                plt.text(x[18],y[6],labels[i], c=color[type],va='center', ha='center', fontsize=12)
        #***************************************   
    
    
    i=i+1



plt.xlabel('Neutrino energy [MeV]', fontsize=14)
plt.ylabel('Flux [$cm^{-2} s^{-1} (MeV^{-1})$]', fontsize=14)
plt.xlim(0.05, 25) 
plt.ylim(1e-2, 1e12)  
plt.yscale('log')
plt.xscale('log')
plt.title('Solar neutrinos fluxes',fontsize=15)



#plt.savefig('output.png', dpi=300)
#plt.savefig('solar_neutrino_flux.pdf')
plt.show()