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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# open data (features_final.csv)
data = pd.read_csv('my_programs/features_final.csv')

# split the data in half
data1 = data.iloc[:len(data)//2]
data2 = data.iloc[len(data)//2:]

# split the data in Y ('Job'),  D ('Master') and X (the rest)
Y1 = data1['Job']
D1 = data1['Master']
X1 = data1.drop(['Job', 'Master', "number"], axis=1)

Y2 = data2['Job']
D2 = data2['Master']
X2 = data2.drop(['Job', 'Master', "number"], axis=1)

# convert to numpy arrays
Y1 = np.array(Y1)
D1 = np.array(D1)
X1 = np.array(X1)

Y2 = np.array(Y2)
D2 = np.array(D2)
X2 = np.array(X2)

# convert to torch tensors 
Y1 = torch.tensor(Y1).float()
D1 = torch.tensor(D1).float()
X1 = torch.tensor(X1).float()

Y2 = torch.tensor(Y2).float()
D2 = torch.tensor(D2).float()
X2 = torch.tensor(X2).float()




#### split data Y1, X1, D1 in train and test
# train: 80% of the data
# test: 20% of the data
Y1_train = Y1[:len(Y1)*80//100]
Y1_test = Y1[len(Y1)*80//100:]
D1_train = D1[:len(D1)*80//100]
D1_test = D1[len(D1)*80//100:]
X1_train = X1[:len(X1)*80//100]
X1_test = X1[len(X1)*80//100:]

# define the neural network
class Net(nn.Module):
    
        def __init__(self, input_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)
    
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))        
            x = self.fc3(x)
            return x

# define the loss function
criterion = nn.MSELoss()

# propensity score NN
# the input_size is the number of features in X
prop_score_NN = Net(input_size=len(X1[0]))

# define the optimizer (adam)
optimizer = optim.Adam(prop_score_NN.parameters(), lr=0.001)
patiente = 50
loss_val_min = 100000
# train the propensity score NN using early stop 
def train_with_early_stop(model, 
                            optimizer, 
                            criterion, 
                            X_train, 
                            Y_train, 
                            X_test, 
                            Y_test, 
                            num_epochs=1000, 
                            patience=50):
    loss_val_min = 100000
    for epoch in range(num_epochs):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

        # validation and early stop
        loss_val = criterion(model(X_test), Y_test)

        # print statistics
        print('epoch: ', epoch, ' loss: ', loss.item(), ' loss_val: ', loss_val.item())

        if loss_val < loss_val_min:
            loss_val_min = loss_val
            patience_so_far = patience
            # save best model 
            torch.save(model.state_dict(), 'my_programs/NN.pt')
        else:
            patience_so_far -= 1
            if patience_so_far == 0:
                # load best model
                model.load_state_dict(torch.load('my_programs/NN.pt'))
                break
    print('Finished Training ')
    return model

prop_score_NN = train_with_early_stop(prop_score_NN,
                                optimizer,
                                criterion,
                                X1_train,
                                D1_train,
                                X1_test,
                                D1_test,
                                num_epochs=1000,
                                patience=50)


# now train the outcome NN
# define the neural network

# outcome NN
# the input_size is the number of features in X plus D (+1)
outcome_NN = Net(input_size=len(X1[0])+1)

# define the optimizer (adam)
optimizer = optim.Adam(outcome_NN.parameters(), lr=0.001)
patiente = 50
loss_val_min = 100000

# the input data hast to be X1 and D1
# X1.shape() is (n, 512) and D1.shape() is (n)
X1_D1_train = torch.cat((X1_train, D1_train.reshape(len(D1_train), 1)), 1)
X1_D1_test = torch.cat((X1_test, D1_test.reshape(len(D1_test), 1)), 1)
# train the outcome NN using early stop
outcome_NN = train_with_early_stop(outcome_NN,
                        optimizer,
                        criterion,
                        X1_D1_train,
                        Y1_train,
                        X1_D1_train,
                        Y1_test,
                        num_epochs=1000,
                        patience=50)

# now we have the propensity score NN and the outcome NN
# we can estimate the average treatment effect
# we will use the test data (X2, D2, Y2)

# we need a neyman orthogonal formula for the ATE, so we will use the 
# inverse probability weighting (IPW) estimator

# calculate the propensity score for each observation in X2
# we will use the propensity score NN
prop_score_NN.eval()
prop_score_NN = prop_score_NN.float() # convert to float
prop_score = prop_score_NN(X2) # calculate the propensity score

# calculate the outcome for each observation in X2
# we will use the outcome NN
outcome_NN.eval()
outcome_NN = outcome_NN.float() # convert to float
X2_D2 = torch.cat((X2, D2.reshape(len(D2), 1)), 1)# concatenate X2 and D2
outcome = outcome_NN(X2_D2) # calculate the outcome

# calculate the IPW estimator
ATE = torch.mean((D2*Y2)/prop_score - ((1-D2)*Y2)/(1-prop_score))

print('ATE: ', ATE.item())


# confidence interval 




# calculate the true ATE (which by construction is 0.2)
# to do that, we can access the data in the original dataframe


# import to make linear regression
from sklearn.linear_model import LinearRegression

real_data = pd.read_csv('my_programs/features_MSC_JOB.csv')

# with a simple regression of JOBS on all other columns, we can calculate the ATE

# confirm that E(Y|master=1, Smiling=0) = 50%
print("confirm:",  real_data[real_data['Master']==1]['Job'].mean())

Y = real_data['Job'].values
X = real_data[['Master','Smiling']].values




# simplest regression
reg = LinearRegression().fit(X, Y)

# print all the coefficients
print(reg.coef_)
# print the intercept
print(reg.intercept_)





