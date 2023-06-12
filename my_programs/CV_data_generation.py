# this program takes the features from the celebA dataset and generates a dataset with
# their features, a binary bool variable that indicates whether they have a master degree or not
# and a binary bool variable that indicates whether they obtained a job or not 

# they will have a master degree depending on some function of their features from celebA
# they will obtain a job depending on some function of their features from celebA 
# and depending on whether they have master degree

# we will be using the align_celeba dataset

import numpy as np
import pandas as pd
# regression packages
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# seed = 241543903
# np.random.seed(seed)

def master_degree(features, male_on_master):
    prob = np.ones(len(features))*0.5
    prob[features['Male'] == 1] -= male_on_master
    return np.random.binomial(1, prob)


def job(features, ATE_truth, male_on_job):
    prob = np.ones(len(features))*0.5
    prob[features['Male'] == 1] -= male_on_job
    prob[features['Master'] == 1] += ATE_truth
    return np.random.binomial(1, prob)


def randomize_data(seed):
    np.random.seed(seed)

    folder = 'C:\\Users\\Roberto\\Desktop\\celebA\\Anno-20230603T133225Z-001\\Anno\\'

    ATE_truth = np.random.uniform(0.1, 0.2)
    male_on_job = np.random.uniform(-0.15, 0.15)
    male_on_master = np.random.uniform(-0.15, 0.15)

    # read the data from the list_attr_celeba.txt file and the other files
    features = pd.read_csv(folder + 'list_attr_celeba.txt', sep='\s+', skiprows=1)
    # binary 0,1 instead of binary -1 ,+1
    features = features.replace(-1, 0)

    features['Master'] = master_degree(features, male_on_master)
    features['Job'] = job(features, ATE_truth, male_on_job)
    features.to_csv('my_programs/features_MSC_JOB.csv', index=True) # to do regressions
    # on features, the index has the number of the image of celebA
    features['number'] = features.index

    # data.csv contains the features for the images of the celebA dataset
    # the index of the images is in the first column. This is a subset of celebA
    image_features = pd.read_csv('my_programs/data.csv', skiprows=1)
    # call the first column 'number', and the other columns will be the features
    image_features.columns = ['number'] + list(range(image_features.shape[1]-1))

    # now merge the two datasets
    number_index2jpg = lambda x: str(x).zfill(6) + '.jpg'
    image_features['number'] = image_features['number'].apply(number_index2jpg)
    features_final = pd.merge(features[["Job", "Master", "number"]], image_features, on='number')

    # save the features in a csv file
    features_final.to_csv('my_programs/features_final.csv', index=False)

    return ATE_truth, male_on_job, male_on_master

def simple_regression():
    data = pd.read_csv('my_programs/features_MSC_JOB.csv')
    Y = data['Job']
    T = data['Master']
    C = data['Smiling']

    # TODO: confirm that you merged properly. Check images 
    
    # regress Y on T and C
    X = data[['Master', 'Smiling']]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    # print the coefficients
    print(model.params)


randomize_data(9317)
simple_regression()
