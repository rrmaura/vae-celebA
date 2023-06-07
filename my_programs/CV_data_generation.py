# this program takes the features from the celebA dataset and generates a dataset with
# their features, a binary bool variable that indicates whether they have a master degree or not
# and a binary bool variable that indicates whether they obtained a job or not 

# they will have a master degree depending on some function of their features from celebA
# they will obtain a job depending on some function of their features from celebA 
# and depending on whether they have master degree

# we will be using the align_celeba dataset

import numpy as np
import pandas as pd

folder = 'C:\\Users\\Roberto\\Desktop\\celebA\\Anno-20230603T133225Z-001\\Anno\\'

# read the data from the list_attr_celeba.txt file and the other files
features = pd.read_csv(folder + 'list_attr_celeba.txt', sep='\s+', skiprows=1)
# maker features are binary 0,1 instead of binary -1 ,+1
features = features.replace(-1, 0)

seed = 1234
np.random.seed(seed)

# def master_degree(features):
#     # the probability of having a master degree is a function of the features
#     # the default probability is 50%
#     # if they are smiling, they have a 10% lower chance of having a master degree
#     # same for young or wearing glasses
#     # if they have Black_Hair, they have a 10% higher chance of having a master degree
#     # same for wearing lipstick or having a pale skin
#     prob = np.ones(len(features))*0.5
#     prob[features['Smiling'] == 1] -= 0.1
#     prob[features['Young'] == 1] -= 0.1
#     prob[features['Eyeglasses'] == 1] -= 0.1
#     prob[features['Black_Hair'] == 1] += 0.1
#     prob[features['Wearing_Lipstick'] == 1] += 0.1
#     prob[features['Pale_Skin'] == 1] += 0.1
#     return np.random.binomial(1, prob)

# def job(features): 
#     # the probability of having a job is a function of the features
#     # the default probability is 50%

#     # FEATURES RELATED TO MSc
#     # if they are smiling, they have a 10% lower chance of having a job
#     # if they have Black_Hair, they have a 10% higher chance of having a job
#     prob = np.ones(len(features))*0.5
#     prob[features['Smiling'] == 1] -= 0.1
#     prob[features['Black_Hair'] == 1] += 0.1

#     # FEATURES UNRELATED TO MSc
#     # if they have a hat, they have a 10% higher chance of having a job
#     # if they have a big lips, they have a 10% lower chance of having a job
#     prob[features['Blond_Hair'] == 1] += 0.1
#     prob[features['Male'] == 1] -= 0.1

#     # if they have a master degree, they have a 20% higher chance of having a job
#     prob[features['Master']] += 0.2
#     return np.random.binomial(1, prob)

# way simpler: 
def master_degree(features):
    prob = np.ones(len(features))*0.5
    prob[features['Male'] == 1] -= 0.1
    return np.random.binomial(1, prob)
def job(features):
    prob = np.ones(len(features))*0.5
    prob[features['Male'] == 1] -= 0.1
    prob[features['Master'] == 1] += 0.1
    return np.random.binomial(1, prob)

features['Master'] = master_degree(features)
features['Job'] = job(features)

# # confirm that prob(master) = prob(master|smiling)*prob(smiling) + prob(master|not smiling)*prob(not smiling)
# pr_smile = np.mean(features['Smiling'])
# pr_not_smile = 1 - pr_smile
# pr_master_smile = np.mean(features[features['Smiling'] == 1]['Master'])
# pr_master_not_smile = np.mean(features[features['Smiling'] == 0]['Master'])
# pr_master = np.mean(features['Master'])
# print("the different probabilities are: ")
# print("prob(smiling) = ", pr_smile)
# print("prob(not smiling) = ", pr_not_smile)
# print("prob(master|smiling) = ", pr_master_smile)
# print("prob(master|not smiling) = ", pr_master_not_smile)
# print("prob(master) = ", pr_master)
# print("prob(master) = ", pr_master_smile*pr_smile + pr_master_not_smile*pr_not_smile)

features.to_csv('my_programs/features_MSC_JOB.csv', index=False)


# open data.csv , which contains the features fo the images of the celebA dataset
# the index of the images is in the first column. Only some of the images are in 
# this second dataset



image_features = pd.read_csv('my_programs/data.csv', skiprows=1)
# call the first column 'index'
image_features.columns = ['number'] + list(range(image_features.shape[1]-1))

# on features, the index is the actual row of the dataset
features['number'] = features.index
# features['number'] = features['number'].astype('int64')

# print the names of the columns of the two datasets
# print(features["number"].head())
# print(image_features["number"].head())

number_index2jpg = lambda x: str(x).zfill(6) + '.jpg'
image_features['number'] = image_features['number'].apply(number_index2jpg)

# merge the two datasets
features_final = pd.merge(features[["Job", "Master", "number"]], image_features, on='number')

# save the features in a csv file
features_final.to_csv('my_programs/features_final.csv', index=False)

