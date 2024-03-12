# Fault Sensitivity Analysis of Printed Bespoke Multilayer Perceptron Classifiers

This github repository is referring to the digital fault analysis as presented in the paper at ETS'24 - Fault Sensitivity Analysis of Printed Bespoke Multilayer Perceptron Classifiers

cite as
```
Fault Sensitivity Analysis of Printed Bespoke Multilayer Perceptron Classifiers
Pal, P.; Afentaki, F.; Zhao, H.; Saglam, G.; Hefenbrock, M.; Zervakis, G.; Beigl, M.; Tahoori, M. B.
2024 IEEE European Test Symposium (ETS), IEEE, 2024
```


## Usage of the code:

### Create the environment

Use the terminal for the following steps:

1. Create the environment from the environment.yml file:

        conda env create -f env.yml


The first line of the yml file sets the new environment's name. In our case the environment name is Qkeras.

2. Activate the new environment: 
    
        conda activate Qkeras

### How to run it?

The code can be simply run by command line through:

~~~
python3 run.py <design>
~~~

where the <design\> is one of the design options in {axAcc, axArgmax, Bespoke, Dropout0.1_axArgmax, Generic}

### Example 

~~~
python3 run.py Bespoke
~~~