# Fault Sensitivity Analysis of Printed Bespoke Multilayer Perceptron Classifiers

This github repository is for the paper at ETS'24 - Fault Sensitivity Analysis of Printed Bespoke Multilayer Perceptron Classifiers

cite as
```
Fault Sensitivity Analysis of Printed Bespoke Multilayer Perceptron Classifiers
Pal, P.; Afentaki, F.; Zhao, H.; Saglam, G.; Hefenbrock, M.; Zervakis, G.; Beigl, M.; Tahoori, M. B.
2024 IEEE European Test Symposium (ETS), IEEE, 2024
```


Usage of the code:

The code can be simply run by command line through:

~~~
$ python3 experiment.py --DATASET 00 --SEED 00 --e_train 0.1 --dropout 0.1 --projectname FaultAnalysis
~~~

where the index of `DATASET` ranges from 00 to 12, `SEED` refers to random seed which was 00 - 09 in the experimental setup, `e_train` is the variation in the variation-aware training which was {0.0, 0.05, 0.1} in the experiment, while `dropout` was choosen from {0.0, 0.05, 0.1} in the experiment as well. The `projectname` would be the name of the folder that stores generated files during training, it can be modified as wanted.

The code for evaluation and visualiztion can be found in e.g., `./FaultAnalysis/Visualization.ipynb` and `./FaultAnalysis/evaluation.ipynb`
