import os
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import natsort
import shutil
from zipfile import ZipFile


sourceDir = '/g/data/q27/jt5911/PostSim'
targetDir = '/scratch/q27/jt5911/SimAnneal/AuPt/'
path2NCPacExe = '/g/data/q27/jt5911/NCPac/NCPac.exe'
path2NCPacInp = '/g/data/q27/jt5911/NCPac/NCPac.inp'


# COPY XYZ FILES TO SINGLE DIRECTORY, RELABEL FILE NUMERICALLY FOLLOWING NATURAL SORT
for dirPath, subDirNames, fileNames in natsort.natsorted(os.walk(sourceDir)):
    for fileName in natsort.natsorted(fileNames):
        if fileName.endswith('S2.zip'):
            with ZipFile(f"{dirPath}/{fileName}", 'r') as f:
                f.extractall(f"{dirPath}/{fileName[-4]/}")
            oriFilePath = f"{dirPath}/{fileName}"
            newFilePath = f"{sourceDir}/{str(counter).zfill(7)}.xyz"
            shutil.copy(oriFilePath, newFilePath)


# PARALLEL BATCH RUN
# 1. Copies executable and input file into each directory containing a searched for file
# 2. Executes the executable in parallel
numThreads = 5  # Total number of thread to use (max should be number of cores on cpu)
numCores = multiprocessing.cpu_count()
if numThreads > numCores:
    print(f"Number of threads too high. CPU cores present: {numCores}")
    
# List of directories for parallel processing
joblist = []
for dirpath, subdirs, files in natsort.natsorted(os.walk(root_dir)):
    for file in files:
        if file.endswith(searchfor):
            joblist.append(dirpath)
            shutil.copy(code_loc1,dirpath)
            shutil.copy(code_loc2,dirpath)
#Run serial jobs on one core           
            os.chdir(dirpath)
            os.system("target.exe")
            print(dirpath)
#def worker(location):
#    os.chdir(location)
#    os.system("target.exe")
    
#def pool_handler():
#    p = Pool(threads)
#    p.map(worker, joblist)

#RUN OPERATION OVER JOBLIST IN PARALLEL
#if __name__ == '__main__':
#    pool_handler() 


#LOOP THROUGH DIRS AND DELETE SPECIFIC FILE
counter = 0
counter2 = 0 
for dirpath, subdirs, files in natsort.natsorted(os.walk(root_dir)):
    foundit = 0
    for file in files:      
        location = os.path.join(dirpath, file)     
        if file.endswith(searchfor):
            location = os.path.join(dirpath, file)  
            counter = counter + 1
#            foundit = 1
            os.remove(location)
            print(counter,location)  
#    if foundit==0:
#        counter2 = counter2 + 1
#        print(counter2,dirpath)
#        else:           
#            os.remove(location)


#CHANGE A LINE IN A FILE TO HEADER
root_dir     = r'C:\Users\opl001\Downloads\DAP\Cu'
searchfor    = '.xyz'
header       = 'CSIRO Nanostructure Databank - Copper Nanoparticle Data Set \n'  #/n for new line

#Loop of files in root directory 
counter = 0
for dirpath, subdirs, files in natsort.natsorted(os.walk(root_dir)):
    for file in files:
        counter = counter + 1
        location = os.path.join(dirpath, file)
        with open(location,'r') as fin:
            data = fin.read().splitlines(True)
            data[1] = header
            print(counter)
        with open(location,'w') as fout:
            fout.writelines(data)  
    


#MERGING LINE FROM SAME TITLED FILES TOGETHER
root_dir     = r'E:\RESEARCH\SHORT\Nanoparticle Datasets\Cu'
searchfor = 'od_FEATURESET.csv'
#Loop of files in root directory 
counter = 0
for dirpath, subdirs, files in natsort.natsorted(os.walk(root_dir)):
    for file in files:   
        if file.endswith(searchfor):
            counter = counter + 1
            location = os.path.join(dirpath, file)
            if counter==1:
                datat = pd.read_csv(location,skiprows=1)
            if counter>1:
                data1 = pd.read_csv(location,skiprows=1)  
                datat = pd.concat([datat,data1],ignore_index=True, sort =False)
datat.to_csv('yaytest.csv',index=False,header=True)                

        

#CHECK IF A FILE EXISTS IN THE RIGHT PLACE
root_dir     = r'C:\Users\opl001\Downloads\Ag'
searchfor = 'od_FEATURESET.csv'

#Loop of files in root directory 
counter = 0
for dirpath, subdirs, files in natsort.natsorted(os.walk(root_dir)):
    foundfile = 0
    for file in files:     
        if file.endswith(searchfor):
            foundfile = 1
            counter = counter + 1
            location = os.path.join(dirpath, file)
    if foundfile==0:
        print(counter,dirpath)    



#EXTRACT VALUE FROM FILES AND OUTPUT THEM INTO A .CSV FILE
root_dir     = r'C:\Users\Human\Downloads\Short\Ni'
searchfor = 'log.lammps'

df = pd.DataFrame(columns=['File','Energy'])
#Loop of files in root directory 
counter = 0
for dirpath, subdirs, files in natsort.natsorted(os.walk(root_dir)):
    for file in files:     
        if file.endswith(searchfor):
            location = os.path.join(dirpath, file)
            with open(location,'r') as file:
                foundit = 0    
                for line in file:                #find line with TotEng and skip to line after it
                    if 'TotEng' in line:   
                        foundit = 1  
                        continue 
                    if foundit==1:
                        value = str.split(line)[3]
                        counter = counter + 1
                        df.loc[counter] = [counter,value] 
                        print(counter,value)
                        break

#write out total file
df.to_csv('energies2.csv',index=False,header=True)  




