from pyDOE import *
import numpy as n

# set of parameters
omega_m = [0.08, 0.17] 
omega_b = [0.021, 0.024]
sigma_8 = [0.7, 1.0] 
h       = [0.5, 0.9] 
ns      = [0.85, 1.00]
#w0      = [-1.5, -0.5], w = -1
#wa      = [-1.5, 1.15] # 

# Euclid emulator : 3 sigma from Planck => move to 7 sigma
omega_m = [0.1306, 0.1546]
omega_b = [0.0215, 0.0235]
sigma_8 = [0.7591, 0.8707]
h       = [0.6155, 0.7307]
ns      = [0.9283, 1.0027]
w0 ∈ [−1.30, −0.70] ,

# eROSITA statistics :
# 100,000 cluster for mass M>2 \times 10^14Msun 
# volume 55 Gpc^3
# more to M>1e13
# smaller volume
# Prediction of expected number of eROSITA cluster per bin of redshift per bin of mass. 
# We deduce accuracy from that
# add cosmology dependence with colossus on these numbers 


# generate the parameters between 0 and 1
n_parameters = 5
n_cosmologies = 200 

# criterion='maximin': maximize the minimum distance between points, but place the point in a randomized location within its interval
grid_maximin = lhs(n_parameters, samples = n_cosmologies, criterion='maximin')

# normalize the parameters to the boundaries specified above
list_omega_m = grid_maximin.T[0] * ( omega_m[1] - omega_m[0] ) + omega_m[0]
list_omega_b = grid_maximin.T[1] * ( omega_b[1] - omega_b[0] ) + omega_b[0]
list_sigma_8 = grid_maximin.T[2] * ( sigma_8[1] - sigma_8[0] ) + sigma_8[0]
list_h       = grid_maximin.T[3] * ( h      [1] - h      [0] ) + h      [0]
list_ns      = grid_maximin.T[4] * ( ns     [1] - ns     [0] ) + ns     [0]
#list_w0      = grid_maximin.T[5] * ( w0     [1] - w0     [0] ) + w0     [0]
#list_wa      = grid_maximin.T[6] * ( wa     [1] - wa     [0] ) + wa     [0]

# save the list
DATA = n.transpose([list_omega_m, list_omega_b, list_sigma_8, list_h, list_ns])
n.savetxt('cosmo_params.ascii', DATA, header = 'omega_m omega_b sigma_8 h ns ')
