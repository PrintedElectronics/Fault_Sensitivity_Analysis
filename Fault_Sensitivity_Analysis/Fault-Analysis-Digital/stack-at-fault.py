import random
import numpy as np
from sklearn.metrics import accuracy_score
import os
from joblib import load
import pandas as pd
import shutil
import fileinput
import sys

def inject_faults(conference, dataset, saf, measures=500):

    filename="gate/top.sv"
    directory=conference+dataset
    designs= [f.path for f in os.scandir(directory) if f.is_dir()]
    parent_folder = os.getcwd()
    
    if saf=="single":
        num_faults=1
    elif saf =="double":
        num_faults=2
    elif saf =="quatre":
        num_faults=4




    for design in designs:

        os.chdir(design)
        if not os.path.exists("gate/top_no_fault.sv"):
            os.system(f"mv {filename} gate/top_no_fault.sv")
        os.system(f"cp gate/top_no_fault.sv {filename}")

        flag=0
        wires=[]
        with open(filename,'r') as file:
            for line in file:
                for word in line.split():
                    if flag==2:
                        wires.append(word[0:-1])
                        if word[-1]==";":
                            flag=0
                            file.seek(0)
                            netlist=file.read()
                            break
                    if word=="wire":
                        flag+=1            


        accuracies=[]
        faults=[]
        fault_values=[]

        for i in range(measures):

            netlist_fault = netlist
            if i>0:
                faulty_wires, fault_value = [], []
                for _ in range(num_faults):
                    faulty_wire=random.choice(wires)
                    fault=random.choice(['1\'b0', '1\'b1'])
                    print(faulty_wire)
                    netlist_fault=netlist_fault.replace(f".Y({faulty_wire})", ".Y()")
                    netlist_fault=netlist_fault.replace(f"({faulty_wire})", f"({fault})")
                    faulty_wires.append(faulty_wire)
                    fault_value.append(int(fault.split("b")[1]))
            else:
                faulty_wires='no fault'
                fault_value='' 

            with open(filename,'w') as file:
                file.write(netlist_fault)

            os.system("make gate_sim")
            y_pred=[]
            y_test=[]
            output_path = 'sim/output.txt'
            sim_output_path = 'sim/sim.Ytest'
            with open(output_path, "r") as file1, open(sim_output_path, "r") as file2:
                for line1, line2 in zip(file1, file2):
                    y_pred.append(int(line1))
                    y_test.append(int(line2))
            acc = accuracy_score(y_test, y_pred)
            print(acc)
            accuracies.append("{:.6}".format(acc*100))
            faults.append(faulty_wires)
            fault_values.append(fault_value)

        res={'fault':faults, 'fault_values':fault_values , "acc": accuracies}
        print(res)
        pd.DataFrame(res).to_excel(saf+'.xlsx')
        os.chdir(parent_folder)

def main():
  
    dataset=sys.argv[1]
    fault=sys.argv[2]
    conference=sys.argv[3]+"/"
  
    measures=3
    inject_faults(conference, dataset, fault, measures)

if __name__=="__main__":
    main()