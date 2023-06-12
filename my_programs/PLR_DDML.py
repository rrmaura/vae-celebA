# Y = outcome = 1 or 0 depending on whether they have a job or not
# D = treatment = 1 or 0 depending on whether they have a master degree or not

# Y = f(X, D) + e
# D = g(X) + e

# split the dataset in half. With half the data, 
# estimate propensity score g(X)
# with half the data, estimate the outcome function f(X, D)

# we will approximate f and g with a neural network with 2 hidden layers


# TODO: should the regression include constant term?? 
# TODO: dropout? 
# TODO: maybe is the t-test instead of 1.96*SE? 

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from CV_data_generation import randomize_data
from sklearn.metrics import mean_squared_error
from scipy.stats import t

# # seeds
# seed = 241543903 
# np.random.seed(seed)
# tf.random.set_seed(seed)

# Define the neural network with relu activation function
# and a sigmoid activation function for the last layer
class Net(tf.keras.Model):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# if the file to save results does not exist, create it and add the header
if not os.path.isfile('my_programs/PLR_results.csv'):
    with open('my_programs/PLR_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'intercept', 'ATE', 'SE', 'CI[0]', 'CI[1]', 'ATE_truth', 'ATE_truth_in_CI', 'male_on_job', 'male_on_master'])

num_simulations = 100

for _ in range(num_simulations): 
    seed = np.random.randint(100000)
    # seed = 9317
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    ATE_truth, male_on_job, male_on_master = randomize_data(seed)
    print("ATE_truth: ", ATE_truth)
    # define the number of folds
    k = 5

    # split the data into k folds
    kf = KFold(n_splits=k, shuffle=True)

    # initialize an array to store the ATE estimates for each fold
    # ate_estimates = np.zeros(k)
    # ate_estimates_simple = np.zeros(k)

    # Open data (features_final.csv)
    data = pd.read_csv('my_programs/features_final.csv')

    # you need to keep track of the double robust estimator for each fold to
    # later calculate the SE of the ATE estimate
    n = data.shape[0]
    Y_noise = np.zeros(n)
    D_noise = np.zeros(n)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        print("Fold: ", i)
        data1, data2 = data.iloc[train_index], data.iloc[test_index]
        # Split the data into Y ('Job'), D ('Master'), and X (the rest)
        Y1 = data1['Job']
        D1 = data1['Master']
        X1 = data1.drop(['Job', 'Master', "number"], axis=1)

        Y2 = data2['Job']
        D2 = data2['Master']
        X2 = data2.drop(['Job', 'Master', "number"], axis=1)

        # Convert to numpy arrays
        Y1 = np.array(Y1)
        D1 = np.array(D1)
        X1 = np.array(X1)

        Y2 = np.array(Y2)
        D2 = np.array(D2)
        X2 = np.array(X2)

        # Define the loss function and the optimizer
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Propensity score NN
        prop_score_NN = Net(input_size=X1.shape[1])

        # compile the model
        prop_score_NN.compile(optimizer=optimizer, loss=loss_fn)

        # Train the propensity score NN using fit(), using the loss function and optimizer defined above
        # and doing early stop with patience=10
        prop_score_NN.fit(X1, D1, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4)])

        # now the same but with outcome NN
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        outcome_NN = Net(input_size=X1.shape[1])
        outcome_NN.compile(optimizer=optimizer, loss=loss_fn)
        outcome_NN.fit(X1, Y1, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])

        #  now we have the propensity score NN and the outcome NN
        # we can estimate the average treatment effect
        # we will use the test data (X2, D2, Y2)

        # first, partial out (Y from X) and (D from X)
        Y_hat = outcome_NN(X2)
        D_hat = prop_score_NN(X2)
        Y_noise[test_index] = Y2 - Y_hat.numpy().flatten()
        D_noise[test_index] = D2 - D_hat.numpy().flatten()


    # calculate the mean ATE estimate across all folds, 
    # by doing OLS of Y_noise on D_noise 
    # Also, obtain the SE of the ATE estimate and CI directly from OLS

    # # fit a linear regression model to estimate the ATE
    # reg = LinearRegression(fit_intercept=False).fit(D_noise.reshape(-1, 1), Y_noise)
    # # calculate the mean ATE estimate across all folds
    # # intercept = reg.intercept_
    # ATE = reg.coef_[0]
    # # calculate the standard error of the ATE estimate
    # Y_noise_pred = reg.predict(D_noise.reshape(-1, 1))
    # SE = np.sqrt(mean_squared_error(Y_noise, Y_noise_pred) / len(Y_noise))
    # # calculate the confidence interval of the ATE estimate
    # CI = (ATE - 1.96 * SE, ATE + 1.96 * SE)


    # higher order neyman orthogonal 
    #second_p_est = np.mean(res_p_first**2)
    mu_2 = np.mean(D_noise ** 2)
    #cube_p_est = np.mean(res_p_first**3) - 3 * np.mean(res_p_first) * np.mean(res_p_first**2)
    mu_3 = np.mean(D_noise ** 3)-3*np.mean(D_noise)*np.mean(D_noise ** 2)
    # mult_p_est = res_p_second**3 - 3 * second_p_est * res_p_second - cube_p_est
    multiplier = D_noise**3 -  3 * mu_2 * D_noise - mu_3
    #     robust_ortho_est_ml = np.mean(res_q * mult_p_est)/np.mean(res_p * mult_p_est)
    ATE = np.mean(Y_noise * multiplier) / np.mean(D_noise * multiplier)
    SE = np.sqrt(np.mean((Y_noise - ATE * D_noise) ** 2) / len(Y_noise))
    CI = (ATE - 1.96 * SE, ATE + 1.96 * SE)


    # # now the same without intercept
    # reg = LinearRegression(fit_intercept=False).fit(D_noise.reshape(-1, 1), Y_noise)
    # ATE2 = reg.coef_[0]
    # Y_noise_pred = reg.predict(D_noise.reshape(-1, 1))
    # SE2 = np.sqrt(mean_squared_error(Y_noise, Y_noise_pred) / len(Y_noise))
    # CI2 = (ATE2 - 1.96 * SE2, ATE2 + 1.96 * SE2)

    print('FINAL ATE: ', ATE)
    
    # append to a CSV file the ATE, the SE, and the confidence interval
    # for the doubly robust estimator and the simple estimator
    # attach the true ATE to the CSV file and whether the true ATE is in the confidence interval
    # for the doubly robust estimator and the simple estimator
    

    ATE_truth_in_CI = (ATE_truth >= CI[0]) and (ATE_truth <= CI[1])
    # save the ATE, SE, and CI to a csv file
    with open('my_programs/PLR_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([seed, 0, ATE, SE, CI[0], CI[1], ATE_truth, ATE_truth_in_CI, male_on_job, male_on_master])
