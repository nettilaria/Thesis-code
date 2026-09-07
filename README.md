1) _SolarFlux.py_ reproduces the solar neutrino flux. Data are taken from https://www.sns.ias.edu/%7Ejnb/

2) _Rate.py_ reproduces the differential rates of CEVNS events w/ and w/o the acceptance. In particular:
   - theoretical formula from https://arxiv.org/pdf/2603.00554
   -  plot to reproduce: https://arxiv.org/pdf/2604.06002

4) _Efficiency.py_ contains the efficiency taken from https://github.com/XENONnT/cevns_data_release/blob/master/plot.ipynb.
   It was then modified in order to obtain the acceptance, because of Fig.1 of https://arxiv.org/pdf/2604.06002.

5) _CEVNS*_ contains the backgrounds of XENONnT.
   The structure of the code is taken from https://github.com/XENONnT/cevns_data_release/blob/master/plot.ipynb
   Then I have:
   -  divided each science run, instead of summing them up;
   -  added the other two backgrounds (Neutron and ER). 
