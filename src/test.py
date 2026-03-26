import numpy as np
import matplotlib.pyplot as plt
from bouquet import GEQDSKEquilibrium # need bouquet as dependency, "pip install bouquet"


fp = '/Users/nelsonlab/codes/Equilibria/test_1.geqdsk'

# Find all parameters here: https://github.com/d-burg/bouquet/blob/377608605628780e553351f1a43d6a598c6d31ca/bouquet/io/geqdsk.py#L769
eq = GEQDSKEquilibrium(fp)
R_grid = eq.R_grid # m, 1D R grid
Z_grid = eq.Z_grid # m, 1D Z grid
psi_RZ = eq.psi_RZ # 2D poloidal flux array at each RZ grid point
pres = eq.pres # pressure evaluated at each psi_N
psi_N = eq.psi_N # Normalized psi
psi_boundary = eq.psi_boundary # poloidal flux at boundary, seems to actually be at the center
R_mag = eq.R_mag # m, R at magnetic axis including finite Beta effects (R0)
a = eq.geometry["a"] # m, minor radius
plasma_surfArea = eq.geometry["surfArea"]
plasma_vol = eq.geometry["vol"]
psi_N_RZ = eq.psi_N_RZ

fig,ax = plt.subplots()
ax.plot(psi_N_RZ[:,0])
plt.show()

# R - AuxQuantities PSIRZ_NORM
# a = fluxSurfaces-geo-a
# plasma_surf_area = fluxSurfaces-geo-surfArea
# plasma_vol = fluxSurfaces-geo-vol

# # what is the difference between these two?
# print(R_center) # no shafranov shift (vacuum)
# print(R_mag)