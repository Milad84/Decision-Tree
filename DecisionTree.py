#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      milad
#
# Created:     14/01/2021
# Copyright:   (c) milad 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# save filepath to variable for easier access
melbourne_file_path = 'E:/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data

print (melbourne_data.describe())

print (melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# We'll use the dot notation to select the column we want to predict, which is
#called the prediction target. By convention, the prediction target is called y.
# So the code we need to save the house prices in the Melbourne data is

y = melbourne_data.Price

##The columns that are inputted into our model (and later used to make predictions)
##are called "features." In our case, those would be the columns used to determine
##the home price. Sometimes, you will use all columns except the target as features.
##Other times you'll be better off with fewer features.
##
##For now, we'll build a model with only a few features. Later on you'll see how to
##iterate and compare models built with different features.
# We select multiple features by providing a list of column names inside brackets.
# Each item in that list should be a string (with quotes).

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print(X.describe())

# Define model. Specify a number for random_state to ensure same results each run

melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

DecisionTreeRegressor(random_state=1)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

##Many people make a huge mistake when measuring predictive accuracy.
## They make predictions with their training data and compare those predictions
## to the target values in the training data.
## You'll see the problem with this approach and how to solve it in a moment,
##  but let's think about how we'd do this first.

##You'd first need to summarize the model quality into an understandable way.
##If you compare predicted and actual home values for 10,000 houses,
##you'll likely find mix of good and bad predictions. Looking through a list of
##10,000 predicted and actual values would be pointless. We need to summarize this
##into a single metric. There are many metrics for summarizing model quality,
##but we'll start with one called Mean Absolute Error (also called MAE).
##Let's break down this metric starting with the last word, error.
##
##The prediction error for each house is:

# error=actual−predicted

#MAE

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
error = mean_absolute_error (y, predicted_home_prices)

print (error)

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


### a tree's depth is a measure of how many splits it makes before coming to a prediction.
### This is a relatively shallow tree
### When we divide the houses amongst many leaves, we also have fewer houses in each leaf.
## Leaves with very few houses will make predictions that are quite close to those homes' actual values,
## but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).

##This is a phenomenon called overfitting, where a model matches the training data
##almost perfectly, but does poorly in validation and other new data. On the flip side,
## if we make our tree very shallow, it doesn't divide up the houses into very distinct groups.


##There are a few alternatives for controlling the tree depth, and many allow for
## some routes through the tree to have greater depth than other routes. But the
## max_leaf_nodes argument provides a very sensible way to control overfitting vs
## underfitting. The more leaves we allow the model to make, the more we move from
## the underfitting area in the above graph to the overfitting area.
##
##We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:




def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

##    We can use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
   
