# this program opens my CSV file and plots an histogram and a 
# standard normal distribution on top of it to compare. 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats



# read the data from the CSV file
# data = pd.read_csv('C:/Users/Roberto/Desktop/results_to_plot.csv')
data = pd.read_csv('my_csv/results_complicated.csv')

# get the columns ATE, ATE_truth and SE from the data
ATE = data['ATE']
ATE_truth = data['ATE_truth']
SE = data['SE']

# how many units are there in the data
n = len(ATE)
print('n = ', n)

# 
ATE_in_CI = data['ATE_in_CI']
print('coverage = ', np.sum(ATE_in_CI)/n)

# plot (ATE - ATE_truth)/SE and a standard normal distribution on top of it 

plt.hist((ATE - ATE_truth)/SE, bins=50, density=True)
mu = 0
sigma = 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.title('Histogram of (ATE - ATE_truth)/SE')
plt.xlabel('(ATE - ATE_truth)/SE')
plt.ylabel('Density')
plt.grid(True)
plt.show()
