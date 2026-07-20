Hi! This repository is an implementation of the tokamak pedestal model outlined in https://doi.org/10.1088/1741-4326/ad4b3e. 
It includes the following features:
- Full non-linear and Picard-solved Saarelma-Connor model, modified to include the full solution to the three fluid system.
- Solver looping between the modified Saarelma-Connor model and EPEDNN to self-consistently predict the temperature and density profile from psi_N=[0.85,1.0].

This code was created by John Anthony Labbate (john.a.labbate@columbia.edu) and Andrew Oak Nelson. Enjoy!
