# Script to truncate excessive lines from LAMMPS log files using vim and sed
# Author: Jonathan Yik Chang Ting
# Date: 6/12/2020

vim -e *.log <<@@@
g/ERROR/?LAMMPS?,/Last command/d
wq
@@@

vim -e *.log <<@@@
g/ERROR/?ERROR?,/Last command/d
wq
@@@

sed -i '/LAMMPS/,$!d' *log
