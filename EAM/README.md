# README file for EAM directory
# Author: Jonathan Yik Chang Ting
# Date: 10/10/2020

The EAM directory contains the EAM alloy potential files for the classical MD simulations and the machineries used to generate them.

The EAM setfl files for alloys are generated in DYNAMO format using the tools in LAMMPS/tools/eam_database provided by 
Xiaowang Zhou (Sandia), xzhou at sandia.gov
based on his paper:
X. W. Zhou, R. A. Johnson, and H. N. G. Wadley, Phys. Rev. B, 69, 144113 (2004)

Alloys involved in this project are:
Au, Pd, Pt, Co
which are all covered by the paper mentioned above.

Instructions:
- Edit the variables at the top of gen_alloy_eam.sh, run as a Bash script.
