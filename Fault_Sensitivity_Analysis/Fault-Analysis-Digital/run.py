import os
import sys 

if __name__=="__main__":

    directory="Designs/ETS_"+sys.argv[1]

    subfolders= [f.path for f in os.scandir(directory) if f.is_dir()]
    for folder in subfolders:
        dataset = folder.split("/")[2]
        os.system(f"./runWrapper.sh {dataset} {directory}") 