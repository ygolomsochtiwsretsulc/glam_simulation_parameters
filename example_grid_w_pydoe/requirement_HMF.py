import numpy as np
import pyccl as ccl
from pyDOE import *
import colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function

cosmo_planck18 = ccl.Cosmology(
    Omega_c = 0.11933/0.6766**2, # 0.26066
    Omega_b = 0.02242/0.6766**2, # 0.04897
    h = 0.6766,
    n_s = 0.9665,
    sigma8 = None,
    A_s = np.e**(3.047)/10**10, # 2.1052e-09
    Omega_k = 0.0,
    Omega_g = None,
    Neff = 3.046, # standard model of particle physics
    m_nu = 0.0, # in eV
    m_nu_type = None, # 'inverted', 'normal', 'equal' or 'list'
    w0 = -1.0,
    wa = 0.0,
    T_CMB = ccl.physical_constants.T_CMB, # 2.725
    bcm_log10Mc=14.079181246047625, # BCM baryon correction model on PK https://arxiv.org/abs/1510.06034
    bcm_etab=0.5,                   # BCM
    bcm_ks=55.0,                    # BCM
    mu_0=0.0,                       # modified gravity model parameter
    sigma_0=0.0,                    # modified gravity model parameter
    z_mg=None,                      # modified growth rate parameter
    df_mg=None,                     # modified growth rate parameter
    transfer_function='boltzmann_camb',  # for power spectrum analysis
    matter_power_spectrum='halofit',     # for power spectrum analysis
    baryons_power_spectrum='nobaryons',  # for power spectrum analysis
    mass_function='tinker10',            # for mass function analysis
    halo_concentration='duffy2008',      # for mass function analysis
    emulator_neutrinos='strict',         # for power spectrum analysis
)
cosmo = cosmo_planck18

# array of redshifts
zs = np.arange(0.1, 1, 0.1)
# compute comoving radial distance, Mpc (no little h)
dC = ccl.comoving_radial_distance(cosmo, 1/(1+zs))
print(dC)

# compute comoving radial distance, Mpc (no little h)
dC_ang = ccl.comoving_angular_distance(cosmo, 1/(1+zs))
print(dC_ang)

# in curved cosmo, these are different
z = 1.0
curved_cosmo = ccl.Cosmology(Omega_k = 0.1, Omega_c=0.17, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
chi_rad  = ccl.comoving_radial_distance(curved_cosmo, 1/(1+z))
chi_curved = ccl.comoving_angular_distance(curved_cosmo, 1/(1+z))
print(chi_rad-chi_curved)

# luminosity distance, Mpc
dL = ccl.luminosity_distance(cosmo, 1/(1+zs))
print(dL)
# distance modulus = 5 * log(luminosity distance / 10 pc). m = M + distance modulus, where m is the apparent magnitude and M is the absolute magnitude.
DM = ccl.distance_modulus(cosmo, 1/(1+zs))
print(DM)

# Compute Scale factor, a, at a comoving radial distance chi.
a_exp = ccl.scale_factor_of_chi(cosmo, dC)
z_inferred = 1/a_exp-1 
print(a_exp)
print(z_inferred)
print(z_inferred-zs)

# for a flat cosmology, Omega_k=0
# comoving volume for the full sky up to redshift z is given in Mpc^3 by
comoving_volume = 4 * np.pi * dC**3. / 3.
print(comoving_volume*1e-9, 'Gpc^3')

# for 13500 deg2 and redshift range 0.1 - 0.8 (like eROSITA) the volume is 
area_fraction = 13500 * np.pi / 129600.
comoving_volume_08 = 4 * np.pi * ccl.comoving_radial_distance(cosmo, 1/(1+0.8))**3. / 3.
comoving_volume_01 = 4 * np.pi * ccl.comoving_radial_distance(cosmo, 1/(1+0.1))**3. / 3.
comoving_volume = (comoving_volume_08 - comoving_volume_01)*area_fraction
print(comoving_volume*1e-9, 'Gpc^3') # 32.29043507433001 Gpc^3

def get_volume(zmin, zmax, area):
	area_fraction = area * np.pi / 129600.
	comoving_volume_min = 4 * np.pi * ccl.comoving_radial_distance(cosmo, 1/(1+zmin))**3. / 3.
	comoving_volume_max = 4 * np.pi * ccl.comoving_radial_distance(cosmo, 1/(1+zmax))**3. / 3.
	return 1e-9 * (comoving_volume_max - comoving_volume_min)*area_fraction


import os 
import astropy.io.fits as fits
data = fits.open(os.path.join(os.environ['MD10'], 'MD10_eRO_CLU_b8_CM_0_pixS_20.0_wCount_June2020.fit'))[1].data

area = 34089./2.
exgal_erositaDE_area = ( abs(data['g_lat'])>10 ) & ( abs(data['g_lon'])>180 )
detected = (data['COUNTS_05_20_1p0_r500c_CLU_self']>50) & (data['HALO_M200c']>1e14) & (exgal_erositaDE_area)
mass = np.log10(data['HALO_M200c']) [detected]
zz = data['redshift_R'][detected]

dlogM = 0.2
dZ = 0.1
bins_M = np.arange( 14.0, mass.max() + dlogM, dlogM )
bins_Z = np.arange( 0., 1.5, dZ )
HH = np.histogram2d( zz, mass, bins = [bins_Z, bins_M])[0]

density = HH.sum()/area
print(HH.sum())
print(density)

volumes = np.array([ get_volume(zmin, zmax, area = area) for zmin, zmax in zip(bins_Z[:-1], bins_Z[1:]) ])

zmins = np.round(bins_Z[:-1],1).astype('str')
zmaxs = np.round(bins_Z[1:] ,1).astype('str')
Mmins = np.round(bins_M[:-1],1).astype('str')
Mmaxs = np.round(bins_M[1:] ,1).astype('str')
M_values = (bins_M[:-1]+bins_M[1:])/2.
Z_values = (bins_Z[:-1]+bins_Z[1:])/2.

print('\\hline')
print(" M min & M max & V & ", " & ".join(zmins), " \\\\")
print(" M sun & M sun & Gpc3 & ", " & ".join(zmaxs), " \\\\")
print('\\hline')

for el, m0, m1, Vi in zip(HH.T, Mmins, Mmaxs, volumes):
	print(m0, " & ", m1, " & ", np.round(Vi,1), " & ", " & ".join(el.astype('int').astype('str')), " \\\\")

print('\\hline')
print('\\hline')
for el, m0, m1, Vi in zip(HH.T, Mmins, Mmaxs,volumes):
	print(m0, " & ", m1, " & ", np.round(Vi,1), " & "," & ".join(np.round((100*el**-0.5),2).astype('str')), " \\\\")

print('\\hline')
print('\\hline')

## Impact of cosmological parameters variation
# generate the parameters between 0 and 1
n_parameters = 5
n_cosmologies = 500 

# Euclid emulator : 3 sigma from Planck => move to 7 sigma
omega_m = [0.1306, 0.1546]
omega_b = [0.0215, 0.0235]
sigma_8 = [0.7591, 0.8707]
h       = [0.6155, 0.7307]
ns      = [0.9283, 1.0027]

sigma_omega_m = ( omega_m[1] - omega_m[0] )/7.
sigma_omega_b = ( omega_b[1] - omega_b[0] )/7.
sigma_sigma_8 = ( sigma_8[1] - sigma_8[0] )/7.
sigma_h       = ( h      [1] - h      [0] )/7.
sigma_ns      = ( ns     [1] - ns     [0] )/7.

mean_omega_m  = 0.14175
mean_omega_b  = 0.02242
mean_sigma_8  = 0.8149
mean_h        = 0.6766
mean_ns       = 0.9665

print( 'omega_m =', mean_omega_m , '$\pm$ 3 (5, 7)', sigma_omega_m )
print( 'omega_b =', mean_omega_b , '$\pm$ 3 (5, 7)', sigma_omega_b )
print( 'sigma_8 =', mean_sigma_8 , '$\pm$ 3 (5, 7)', sigma_sigma_8 )
print( 'h       =', mean_h       , '$\pm$ 3 (5, 7)', sigma_h       )
print( 'ns      =', mean_ns      , '$\pm$ 3 (5, 7)', sigma_ns      )

omega_m = [mean_omega_m - 3 * sigma_omega_m , mean_omega_m + 3 * sigma_omega_m]
omega_b = [mean_omega_b - 3 * sigma_omega_b , mean_omega_b + 3 * sigma_omega_b]
sigma_8 = [mean_sigma_8 - 3 * sigma_sigma_8 , mean_sigma_8 + 3 * sigma_sigma_8]
h       = [mean_h       - 3 * sigma_h       , mean_h       + 3 * sigma_h      ]
ns      = [mean_ns      - 3 * sigma_ns      , mean_ns      + 3 * sigma_ns     ]


# criterion='maximin': maximize the minimum distance between points, but place the point in a randomized location within its interval
grid_maximin = lhs(n_parameters, samples = n_cosmologies, criterion='maximin')

# normalize the parameters to the boundaries specified above
list_omega_m = grid_maximin.T[0] * ( omega_m[1] - omega_m[0] ) + omega_m[0]
list_omega_b = grid_maximin.T[1] * ( omega_b[1] - omega_b[0] ) + omega_b[0]
list_sigma_8 = grid_maximin.T[2] * ( sigma_8[1] - sigma_8[0] ) + sigma_8[0]
list_h       = grid_maximin.T[3] * ( h      [1] - h      [0] ) + h      [0]
list_ns      = grid_maximin.T[4] * ( ns     [1] - ns     [0] ) + ns     [0]

MANY_HMF = []
DATA = np.transpose([list_omega_m, list_omega_b, list_sigma_8, list_h, list_ns])
for el in DATA :
	Omega_m = el[0]/el[3]**2
	Omega_b = el[1]/el[3]**2
	my_cosmo = {'flat': True, 'H0': el[3]*100., 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8': el[2], 'ns': el[4]}
	cosmology.setCosmology('my_cosmo', my_cosmo)
	MANY_HMF.append( np.array([ mass_function.massFunction(M_values, z_i, mdef = '200c', model = 'tinker08', q_out = 'dndlnM', q_in='M') for z_i in Z_values]) )

MANY_HMF = np.array( MANY_HMF )

mean_hmfs = np.mean(MANY_HMF, axis = 0 )
min_hmfs = np.min(MANY_HMF, axis = 0 )
max_hmfs = np.max(MANY_HMF, axis = 0 )

frac_hmfs = np.round(100*MANY_HMF.std(axis=0)/MANY_HMF.mean(axis=0),2)
frac_hmfs = np.round(100*(max_hmfs-min_hmfs)/MANY_HMF.mean(axis=0),2)
for el, m0, m1, Vi in zip(frac_hmfs.T, Mmins, Mmaxs,volumes):
	print(m0, " & ", m1, " & ", np.round(Vi,1), " & "," & ".join(np.round(el,2).astype('str')), " \\\\")

print('\\hline')
print('\\hline')

#hmd_200c = ccl.halos.MassDef200c()
#for el in DATA :
	#Omega_c = (el[0]-el[1])/el[3]**2
	#Omega_b = el[1]/el[3]**2
	#cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b,h=el[3], sigma8=el[2], n_s=el[4])
	#hmfs = (ccl.halos.MassFuncTinker10(cosmo))
	#hmf_200m.get_mass_function(cosmo, M_values, 1./(1+Z_values))
	#nm = mf.get_mass_function(cosmo, m_arr, 1.)


# 5 sigma list 

omega_m = [mean_omega_m - 5 * sigma_omega_m , mean_omega_m + 5 * sigma_omega_m]
omega_b = [mean_omega_b - 5 * sigma_omega_b , mean_omega_b + 5 * sigma_omega_b]
sigma_8 = [mean_sigma_8 - 5 * sigma_sigma_8 , mean_sigma_8 + 5 * sigma_sigma_8]
h       = [mean_h       - 5 * sigma_h       , mean_h       + 5 * sigma_h      ]
ns      = [mean_ns      - 5 * sigma_ns      , mean_ns      + 5 * sigma_ns     ]

# criterion='maximin': maximize the minimum distance between points, but place the point in a randomized location within its interval
grid_maximin = lhs(n_parameters, samples = n_cosmologies, criterion='maximin')

# normalize the parameters to the boundaries specified above
list_omega_m = grid_maximin.T[0] * ( omega_m[1] - omega_m[0] ) + omega_m[0]
list_omega_b = grid_maximin.T[1] * ( omega_b[1] - omega_b[0] ) + omega_b[0]
list_sigma_8 = grid_maximin.T[2] * ( sigma_8[1] - sigma_8[0] ) + sigma_8[0]
list_h       = grid_maximin.T[3] * ( h      [1] - h      [0] ) + h      [0]
list_ns      = grid_maximin.T[4] * ( ns     [1] - ns     [0] ) + ns     [0]


MANY_HMF7 = []
DATA = np.transpose([list_omega_m, list_omega_b, list_sigma_8, list_h, list_ns])
for el in DATA :
	Omega_m = el[0]/el[3]**2
	Omega_b = el[1]/el[3]**2
	my_cosmo = {'flat': True, 'H0': el[3]*100., 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8': el[2], 'ns': el[4]}
	cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
	MANY_HMF7.append( np.array([ mass_function.massFunction(M_values, z_i, mdef = '200c', model = 'tinker08', q_out = 'dndlnM', q_in='M') for z_i in Z_values]) )

MANY_HMF7 = np.array( MANY_HMF7 )

mean_hmfs7 = np.mean(MANY_HMF7, axis = 0 )
min_hmfs7 = np.min(MANY_HMF7, axis = 0 )
max_hmfs7 = np.max(MANY_HMF7, axis = 0 )

frac_hmfs7 = np.round(100*MANY_HMF7.std(axis=0)/MANY_HMF7.mean(axis=0),2)
frac_hmfs7 = np.round(100*(max_hmfs7-min_hmfs7)/MANY_HMF.mean(axis=0),2)
for el, m0, m1, Vi in zip(frac_hmfs7.T, Mmins, Mmaxs,volumes):
	print(m0, " & ", m1, " & ", np.round(Vi,1), " & "," & ".join(np.round(el,2).astype('str')), " \\\\")

print('\\hline')
print('\\hline')

# 7 sigma list 

omega_m = [mean_omega_m - 7 * sigma_omega_m , mean_omega_m + 7 * sigma_omega_m]
omega_b = [mean_omega_b - 7 * sigma_omega_b , mean_omega_b + 7 * sigma_omega_b]
sigma_8 = [mean_sigma_8 - 7 * sigma_sigma_8 , mean_sigma_8 + 7 * sigma_sigma_8]
h       = [mean_h       - 7 * sigma_h       , mean_h       + 7 * sigma_h      ]
ns      = [mean_ns      - 7 * sigma_ns      , mean_ns      + 7 * sigma_ns     ]

# criterion='maximin': maximize the minimum distance between points, but place the point in a randomized location within its interval
grid_maximin = lhs(n_parameters, samples = n_cosmologies, criterion='maximin')

# normalize the parameters to the boundaries specified above
list_omega_m = grid_maximin.T[0] * ( omega_m[1] - omega_m[0] ) + omega_m[0]
list_omega_b = grid_maximin.T[1] * ( omega_b[1] - omega_b[0] ) + omega_b[0]
list_sigma_8 = grid_maximin.T[2] * ( sigma_8[1] - sigma_8[0] ) + sigma_8[0]
list_h       = grid_maximin.T[3] * ( h      [1] - h      [0] ) + h      [0]
list_ns      = grid_maximin.T[4] * ( ns     [1] - ns     [0] ) + ns     [0]


MANY_HMF7 = []
DATA = np.transpose([list_omega_m, list_omega_b, list_sigma_8, list_h, list_ns])
for el in DATA :
	Omega_m = el[0]/el[3]**2
	Omega_b = el[1]/el[3]**2
	my_cosmo = {'flat': True, 'H0': el[3]*100., 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8': el[2], 'ns': el[4]}
	cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)
	MANY_HMF7.append( np.array([ mass_function.massFunction(M_values, z_i, mdef = '200c', model = 'tinker08', q_out = 'dndlnM', q_in='M') for z_i in Z_values]) )

MANY_HMF7 = np.array( MANY_HMF7 )

mean_hmfs7 = np.mean(MANY_HMF7, axis = 0 )
min_hmfs7 = np.min(MANY_HMF7, axis = 0 )
max_hmfs7 = np.max(MANY_HMF7, axis = 0 )

frac_hmfs7 = np.round(100*MANY_HMF7.std(axis=0)/MANY_HMF7.mean(axis=0),2)
frac_hmfs7 = np.round(100*(max_hmfs7-min_hmfs7)/MANY_HMF.mean(axis=0),2)
for el, m0, m1, Vi in zip(frac_hmfs7.T, Mmins, Mmaxs,volumes):
	print(m0, " & ", m1, " & ", np.round(Vi,1), " & "," & ".join(np.round(el,2).astype('str')), " \\\\")

print('\\hline')
print('\\hline')
