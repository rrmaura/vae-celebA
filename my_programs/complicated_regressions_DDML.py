# Y = outcome = 1 or 0 depending on whether they have a job or not
# D = treatment = 1 or 0 depending on whether they have a master degree or not

# Y = f(X, D) + e
# D = g(X) + e

# split the dataset in half. With half the data, 
# estimate propensity score g(X)
# with half the data, estimate the outcome function f(X, D)

# we will approximate f and g with a neural network with 2 hidden layers

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from CV_data_generation import randomize_data
n_folds = 5
patience = 4
hidden_size = 16
random_param = False # if true, the parameters from the DGP are random. O.w. they are fixed at 0.1
save_output = False
# # seeds
# seed = 241543903 
# np.random.seed(seed)
# tf.random.set_seed(seed)

# Define the neural network with relu activation function
# and a sigmoid activation function for the last layer
class Net(tf.keras.Model):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# if the file to save results does not exist, create it and add the header
if not os.path.isfile('my_csv/results_complicated.csv'):
    with open('my_csv/results_complicated.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'ATE', 'SE', 'CI_lower', 'CI_upper', 'ATE2', 'SE2', 'CI2_lower', 'CI2_upper', 'ATE_truth', 'ATE_in_CI', 'ATE2_in_CI'])

num_simulations = 1

# ATE_truth, male_on_job, male_on_master = randomize_data(seed=42, random_param=random_param)
ATE_truth, male_on_job, male_on_master = randomize_data(42, random_param = False, complicated = True)
# Open data (features_final.csv)
data_original = pd.read_csv('my_csv/features_final.csv')
for _ in range(num_simulations): 
    seed = np.random.randint(200000)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print("ATE_truth: ", ATE_truth)
    
    # split the data into k folds
    kf = KFold(n_splits=n_folds, shuffle=True)

    # initialize an array to store the ATE estimates for each fold
    # ate_estimates = np.zeros(k)
    # ate_estimates_simple = np.zeros(k)

    # take a random 1% of the data with resampling
    # data = data_original.sample(frac=0.1, replace=True, random_state=seed)
    data = data_original

    # data = data.sample(frac=0.1, random_state=seed)

    # you need to keep track of the double robust estimator for each fold to
    # later calculate the SE of the ATE estimate
    n = data.shape[0]
    DR_estimator = np.zeros(n)
    simple_estimator = np.zeros(n)
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
        prop_score_NN.fit(X1, D1, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)])

        # now the same but with outcome NN
        loss_fn = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        outcome_NN = Net(input_size=X1.shape[1]+1)
        outcome_NN.compile(optimizer=optimizer, loss=loss_fn)
        outcome_NN.fit(np.concatenate((X1, D1.reshape(-1, 1)), axis=1), Y1, epochs=100, validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)])

        #  now we have the propensity score NN and the outcome NN
        # we can estimate the average treatment effect
        # we will use the test data (X2, D2, Y2)

        # we need a neyman orthogonal formula for the ATE, so we will use the 
        # double robust estimator

        # calculate the propensity score and predicted outcome for each observation in X2
        propensity_score = prop_score_NN(X2).numpy().flatten()
        predicted_outcome_1 = outcome_NN(np.concatenate((X2, np.ones((X2.shape[0], 1))), axis=1)).numpy().flatten()
        predicted_outcome_0 = outcome_NN(np.concatenate((X2, np.zeros((X2.shape[0], 1))), axis=1)).numpy().flatten()

        # clip the propensity score to be between 0.05 and 0.95
        propensity_score = np.clip(propensity_score, 0.05, 0.95)

        # calculate the IPW estimator

        # ATE = np.mean((D2 * predicted_outcome_1 / propensity_score) - ((1 - D2) * predicted_outcome_0 / (1 - propensity_score)))
        double_robust = (predicted_outcome_1-predicted_outcome_0) + (Y2-predicted_outcome_1)*D2/propensity_score - (Y2-predicted_outcome_0)*(1-D2)/(1-propensity_score)
        ate_estimate = np.mean(double_robust)
        DR_estimator[test_index] = double_robust

        # keep track of the doubly robust estimator for each observation
        
        # try simple estimator, i.e. mean f(X, D) - mean f(X, 1-D)
        ate_estimates_simple = np.mean(predicted_outcome_1-predicted_outcome_0)
        simple_estimator[test_index] = predicted_outcome_1-predicted_outcome_0
        print('ATE: ', ate_estimate)
        print('ATE2: ', ate_estimates_simple)

    # # calculate the mean ATE estimate across all folds
    ATE = np.mean(DR_estimator)
    print('FINAL ATE: ', ATE)

    ATE2 = np.mean(simple_estimator)
    print('FINAL ATE2: ', ATE2)

    # calculate the standard error of the ATE estimate
    V = np.mean((DR_estimator- ATE)**2)
    SE = (V/n)**0.5
    CI = [ATE - 1.96*SE, ATE + 1.96*SE]
    print('SE: ', SE)
    print('CI: ', CI)

    V2 = np.mean((simple_estimator- ATE2)**2)
    SE2 = (V2/n)**0.5
    CI2 = [ATE2 - 1.96*SE2, ATE2 + 1.96*SE2]
    print('SE2: ', SE2)
    print('CI2: ', CI2)

    # save the simple_estimator and the DR_estimator to a csv file
    np.savetxt("my_csv/simple_estimator.csv", simple_estimator, delimiter=",")
    np.savetxt("my_csv/DR_estimator.csv", DR_estimator, delimiter=",")

    # append to a CSV file the ATE, the SE, and the confidence interval
    # for the doubly robust estimator and the simple estimator
    # attach the true ATE to the CSV file and whether the true ATE is in the confidence interval
    # for the doubly robust estimator and the simple estimator

    ATE_truth_in_CI = (ATE_truth >= CI[0]) and (ATE_truth <= CI[1])
    ATE_truth_in_CI = int(ATE_truth_in_CI)
    ATE_truth_in_CI2 = (ATE_truth >= CI2[0]) and (ATE_truth <= CI2[1])
    ATE_truth_in_CI2 = int(ATE_truth_in_CI2)
    # save the ATE, SE, and CI to a csv file
    if save_output:
        with open('my_csv/results_complicated.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([seed, ATE, SE, CI[0], CI[1], ATE2, SE2, CI2[0], CI2[1], ATE_truth, ATE_truth_in_CI, ATE_truth_in_CI2])
