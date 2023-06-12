import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.stats as stats

# open results_CI.csv
results = pd.read_csv('my_programs/results_CI.csv')

plt.hist(results['ATE'] - results['ATE_truth'], bins=100)
plt.show()

# create a normal distribution with mean 0 and standard deviation equal to the standard deviation of the ATE - ATE_truth
# plot the normal distribution on top of the histogram
mu = 0
sigma = np.std(results['ATE'] - results['ATE_truth'])
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, 100000*stats.norm.pdf(x, mu, sigma))
plt.hist(results['ATE'] - results['ATE_truth'], bins=100)
plt.show()



