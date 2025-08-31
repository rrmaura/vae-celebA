import numpy as np 
import pandas as pd

# open both simple_estimator.csv and DR_estimator.csv
simple_estimator = pd.read_csv('my_programs/simple_estimator.csv')
DR_estimator = pd.read_csv('my_programs/DR_estimator.csv')
n = simple_estimator.shape[0]
ATE = np.mean(simple_estimator)
print('FINAL ATE: ', ATE)

ATE2 = np.mean(DR_estimator)
print('FINAL ATE2: ', ATE2)

# # calculate the standard error of the ATE estimate
V = np.mean((DR_estimator- ATE)**2)
SE = (V/n)**0.5
CI = [ATE - 1.96*SE, ATE + 1.96*SE]
print('SE: ', SE)
print('CI: ', CI)

V2 = np.mean((simple_estimator- ATE2)**2)**0.5
SE2 = (V2/n)**0.5
CI2 = [ATE2 - 1.96*SE2, ATE2 + 1.96*SE2]
print('SE2: ', SE2)
print('CI2: ', CI2)