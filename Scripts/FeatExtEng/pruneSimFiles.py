import os
import shutil
import tarfile


PROJECT, USER_GROUP_ID, USER_NAME = 'q27', '564', 'jt5911'
sourceDirs = ['CS']  # 'L10', 'L12', 'RCS','RAL', 'CS'
NPsize = 'large'


def pruneSimFiles(sourcePaths, NPsize='large'):
    print("Removing unnecessary files from MD simulation directories...")
    for sourcePath in sourcePaths:
        print(f"  Source path: {sourcePath}")
        for eleSubDir in os.listdir(sourcePath):
            print(f"    Element subdirectory: {eleSubDir}")
            for NPdirTarBall in os.listdir(f"{sourcePath}/{eleSubDir}"):
                NPdirTarBallPath = f"{sourcePath}/{eleSubDir}/{NPdirTarBall}"
                if NPsize == 'large':
                    if os.path.isdir(NPdirTarBallPath):
                        #for NPitem in os.listdir(NPdirTarBallPath):
                        #    if 'S2.log' in NPitem or 'S2.zip' in NPitem or 'S2.tar.gz' in NPitem: continue
                        #    if os.path.isdir(f"{NPdirTarBallPath}/{NPitem}"): shutil.rmtree(f"{NPdirTarBallPath}/{NPitem}")
                        #    else: os.remove(f"{NPdirTarBallPath}/{NPitem}")
                        continue
                    NPdirPath = NPdirTarBallPath[:-7]
                    if not os.path.exists(NPdirPath):
                        try:
                            with tarfile.open(NPdirTarBallPath, 'r') as f: f.extractall(f"{sourcePath}/{eleSubDir}/")
                        except:
                            print(f"{NPdirTarBall[:-7]} tar ball problematic! Skipping for now...")
                            continue
                else: NPdirPath = NPdirTarBallPath
                os.chdir(NPdirPath)
                for NPitem in os.listdir(NPdirPath):
                    if 'S2.log' in NPitem or 'S2.zip' in NPitem or 'S2.tar.gz' in NPitem: continue
                    if os.path.isdir(f"{NPdirPath}/{NPitem}"): shutil.rmtree(f"{NPdirPath}/{NPitem}")
                    else: os.remove(f"{NPdirPath}/{NPitem}")
                if NPsize == 'large': 
                    os.remove(NPdirTarBallPath)
                    print(f"      Nanoparticle {NPdirTarBall[:-7]} done!")
                else: print(f"      Nanoparticle {NPdirTarBall} done!")


if __name__ == '__main__':
    if NPsize == 'small': sourcePaths = [f"/g/data/{PROJECT}/{USER_NAME}/PostSim/{sourceDir}" for sourceDir in sourceDirs]
    else: sourcePaths = [f"/scratch/{PROJECT}/{USER_NAME}/BNP_MDsim/{sourceDir}50+" for sourceDir in sourceDirs]
    pruneSimFiles(sourcePaths, NPsize)
    print("All DONE!")
