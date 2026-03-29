import numpy as np
from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk

eq = read_eqdsk('g192185.02440')
# print(eq.keys())
'''
print(eq['case'])
print(eq['nr'])
print(eq['nz'])
print(eq['rdim'])
print(eq['zdim'])
print(eq['rcentr'])
print(eq['rleft'])
print(eq['zmid'])
print(eq['raxis'])
print(eq['zaxis'])
print(eq['psimag'])
'''

# Load a g-file
rdim = eq['rdim'] # m, 1D R grid
zdim = eq['zdim'] # m, 1D Z grid
rcentr = eq['rcentr'] # m, radius of the center of the plasma
rleft = eq['rleft'] # m, radius of the left edge of the plasma
zmid = eq['zmid'] # m, height of the midplane of the plasma
raxis = eq['raxis'] # m, radius of the magnetic axis
zaxis = eq['zaxis'] # m, height of the magnetic axis
psimag = eq['psimag'] # Wb, poloidal flux at the magnetic axis
psibry = eq['psibry'] # Wb, poloidal flux at the boundary

psi_RZ = eq['psirz'] # 2D poloidal flux array at each RZ grid point
psi_boundary = eq.psi_boundary # poloidal flux at boundary

# Saarelma-Connor model input parameters
S_plasma = eq.geometry["surfArea"] # m^2, total surface area of plasma
V_plasma = eq.geometry["vol"] # m^3, volume enclosed by the plasma per poloidal flux
a = eq.geometry["a"] # m, minor radius
P = eq['pres'] # Pa, pressure evaluated at each psi_N
R0 = eq.R_mag # location of magnetic axis relative to device rotational line of toroidal symmetry
r_mhd = eq.R_grid # physical radius grid from the file that imports MHD parameters
r_sep =  # physical radius of separatrix as a function of theta

# EPEDNN input parameters
betan =        # Normalized beta
Bt =           # Toroidal magnetic field (T)
bcentr = eq['bcentr'] # T, magnetic field at the center of the plasma
# delta =       # Effective triangularity
Ip = eq['ip']          # Plasma current (MA)
# kappa =        # Elongation