# Classification model with Artificial Neural Network
#We are aiming to predict whether clients will stay with the bank or not
#Artificial Neural Networks are slower to compile than machine learning models, but they can reach higher accuracy levels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#PART 1 - PREPROCESSING THE DATA
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #PART 2 - CREATING THE ANN -----------------------------------------------------------
# #Making the Artificial Neural Network
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
#
# #Initialize the ANN
# classifier = Sequential()
# #Layer 1
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #It's 11 because that's the number of our variables
#
# #Layer 2
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
#
# #Adding the output layer
# classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) #LAST LAYER MUST HAVE INPUT 1 AND SIGMOID FUNCTION, but if there's more than one category for the independent variable, we must use "softmax" instead of Sigmoid
#
# #---------------------------------------------------------------------------------------
#
# #Compiling the ANN
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']  ) #we use binary_crossentropy because we only have two possible independent variables,  0 or 1
#
# #Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size=10, nb_epoch=5) #WARNING these are params to be thoroughly fine tuned for greater accuracy. For example we could try 100 epochs

#PART 3 MAKING PREDICTIONS
#Predicting the Test set results
#y_pred = classifier.predict(X_test)

#Making the confusion matrix
#now we change y_pred to show only the positive values...those over 0.5 which means these clients are more likely to leave the bannk than the ones with a score below 0.5. For medical purposes -like identifying a tumor- we should choose a much higher threshold in order to be much more certain in our prediction, but for bank data 0.5 is fine.

# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
# print('.........................')
# np.set_printoptions(threshold=np.nan) #make numpy show all values
# print('This is y_pred: ', y_pred)
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print(cm) #WARNING this will give us a double array in which the sum of the 1st and 4th values are the correct predictions, and the sum of the 2d and 3d values are the total of incorrect predictions
#
#
# ###########################################################################
# #EXERCISE - PREDICTING THE CHANCE OF ONE PARTICULAR CLIENT LEAVING THE BANK
# # Predicting a single new observation
# """Predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40
# Tenure: 3
# Balance: 60000
# Number of Products: 2
# Has Credit Card: Yes
# Is Active Member: Yes
# Estimated Salary: 50000"""
#
# new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# #In order to know what numeric value corresponds to what we must have the initial dataset handy and a printout of the results after encoding the data. This will let us know what value corresponds to 1, 0 or some other value. WARNING: some data, like country "France" is indicated by 2 values (in this case the two "0,0" at the start of the array signify "France"), not just one number, since there were 3 initial values corresponding to 3 countries (France, Spain, Germany) and so the numerical value representing one country will correspond to two numerical values, not one, due to the creating of the dummy variables. The "0.0" at the start of the arrray is only used to include at least a decimal number in the array, to avoid later the warning "input dtype int32 was converted to float64 by StandardScaler."
#
# #PRINTOUTS----------------------------------------------------------------
#
# print("New prediction's numerical response value is", new_prediction)
#
# if new_prediction > 0.5:
#     print('Client has more chances of leaving the bank than staying')
# else:
#     print('Client will stay with the bank')
#
# #------------------------------------------------------------------------



#PART 4 - Evaluating, improving and tuning the ANN
# #Evaluating the ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from keras.models import Sequential
# from keras.layers import Dense
#
# def  build_classifier():
#     classifier = Sequential()
#     classifier.add(Dense(output_dim=6, init='uniform', activation='relu',input_dim=11))
#     classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
#     classifier.add(Dense(output_dim=1, init='uniform',
#                          activation='sigmoid'))
#     classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
#         'accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn= build_classifier, batch_size = 25, nb_epoch = 100)
# accuracies = cross_val_score(estimator= classifier, X = X_train, y = y_train, cv = 10, n_jobs=  -1) #10 is a recommended value. The value "-1" in njobs is used to activate all cores and run all the calculations in parallel
#
# mean = accuracies.mean()
# variance = accuracies.std()
# print('Mean and variance are respectively:', mean, 'and', variance)

#Fine tuning the model with GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def  build_classifier(optimizer): #important to add "optimizer" as parameter of this function
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform',
                         activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        'accuracy'])
    return classifier

classifier = KerasClassifier(build_fn= build_classifier)
parameters = {'batch_size': [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid= parameters,
                           scoring = 'accuracy',
                           cv= 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print('CLASSIFIER HAS FINISHED RUNNING')
print('Best params were:', best_parameters)
print('Best accuracy was:', best_accuracy)