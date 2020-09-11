"""
Written By: Akash Malhotra
Date: Noember 29, 2019

Problem Statement:
A marketing analyst would like to figure out which customers he could expect to buy 
the new eReader and on what time schedule, based on the company’s last release of 
a high-profile digital reader. He would like to segment customers in order to provide 
them the appropriate advertising.
Knowing that there are four possible values of eReader_Adoption: 
[1]Innovator: purchase within the 1st week 
[2]Early Adopter: purchase within 2 or 3 weeks 
[3]Early Majority: purchase > 3weeks but <= 2 months 
[4]Late Majority: purchase after the first two month 
The criteria and suggestions for segmenting customers are as follows: 
[1] Those who are most likely to purchase immediately (predicted innovators) can be 
contacted and encouraged to go ahead and buy as soon as the new product comes out. 
They may even want the option to pre-order the new device.  On the other hand, 
perhaps very little marketing is needed to the predicted innovators, since they 
are predicted to be the most likely to buy the eReader in the first place. 
[2] Those who are probably likely to purchase earlier (often are opinion leaders, 
predicted early adopter) 
[3] Those who are less likely (predicted early majority) might need some persuasion, 
perhaps a free digital book or two with eReader purchase or a discount on digital music 
playable on the new eReader. 
[4] The least likely (predicted late majority), can be marketed to passively, or perhaps 
not at all if marketing budgets are tight and those dollars need to be spent incentivizing 
the most likely customers to buy.
"""

import pandas as pd, numpy as np
from matplotlib.pyplot import plot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data as plot_cm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

#Load the dataset
retail_training = pd.read_csv("Online_Retailer_DataSet_Training.csv")

#Delete missing values
retail_training = retail_training.dropna()

#Obtain input and output vectors
X = retail_training.drop(retail_training.columns[-1], axis=1)
Y = retail_training[retail_training.columns[-1]]

#Convert nominal data types to binary data types
X = pd.get_dummies(X)
X = X.drop(['Gender_M', 'Marital_Status_S', 'Browsed_Electronics_12Mo_No',
            'Bought_Electronics_12Mo_No', 'Bought_Digital_Media_18Mo_No',
            'Bought_Digital_Books_No'], axis=1)

#Build a classifier
clf = RandomForestClassifier(n_estimators=20, max_depth=20,
                             random_state=42)

#Feature selection on training data: determine which features give the best result using
#this classifier
new_X = pd.DataFrame()
best_features = []
best_score = 0
for col in X.columns:
    new_X = pd.concat([new_X, X[col]], axis=1)
    clf.fit(np.array(new_X), np.array(Y))
    y_pred = clf.predict(np.array(new_X))
    score = accuracy_score(np.array(Y), y_pred)
    if score > best_score:
        best_score = score
        best_features = new_X.columns
print("Best score on training data after feature selection: ", best_score)
print("With number of features: ", len(best_features))
new_X = X[best_features]

#Stratified K-Fold Cross Validation
scores = []
skf = StratifiedKFold(n_splits=5) #5 fold cross validation
for train_index, test_index in skf.split(new_X,Y):
    X_train, X_test = np.array(new_X)[train_index], np.array(new_X)[test_index]
    y_train, y_test = np.array(Y)[train_index], np.array(Y)[test_index]
    clf.fit(X_train, y_train)  

    #print(clf.feature_importances_)
    y_pred = np.array(clf.predict(X_test))
    
    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plot_cm(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    print(score)
    scores.append(score)
    clf.feature_importances_

print("Accuracy : ", np.mean(scores), " +- ", np.std(scores))

#Predict the customer classes from retail scoring 
#Read and preprocess the data
retail_scoring = pd.read_csv("Online_Retailer_DataSet_Scoring.csv")
retail_scoring = pd.get_dummies(retail_scoring)
retail_scoring = retail_scoring.drop(['Gender_M', 'Marital_Status_S', 'Browsed_Electronics_12Mo_No',
            'Bought_Electronics_12Mo_No', 'Bought_Digital_Media_18Mo_No',
            'Bought_Digital_Books_No'], axis=1)
    
#Delete the missing values
retail_scoring = retail_scoring.dropna()

#make sure the sequence of columns of this data matches that of training dataset
retail_scoring = retail_scoring[new_X.columns]
retail_scoring.columns = new_X.columns

#predict the result
result = list(clf.predict(np.array(retail_scoring)))

#To do: Make inferences based on Sensitivity and Recall of Confusion Matrix




