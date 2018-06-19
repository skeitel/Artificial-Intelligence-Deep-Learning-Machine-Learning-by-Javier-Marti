#This program analyzes the Iris machine learning database and applies machine learning techniques to browse the data inside it, evaluate the accuracy of several methods of pronostication, and then split and test the data with 25 different combinations of the K nearest neighbor technique, to ensure maximum accuracy of prediction.
#Machine learning tutorial https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4
from sklearn.datasets import load_iris

iris = load_iris()

#print the data in the dataset
print('\n This is iris.data \n' , iris.data)
#print the names of the four features
print('\n This is iris.feature_names \n', iris.feature_names)
#print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
#print integers representing the species of each observation
print('\n This is iris.target \n' , iris.target)
#print names of targets
print('\n This is iris.target_names \n', iris.target_names)
#print the type of features and response
print('\n This is the type(iris.data) \n' , type(iris.data))
print('\n This is type(iris.target) \n' , type(iris.target))

## IMPORTANT / to use scikit learn: 1) data must be numerical 2) in the shape of numpy arrays

#print shape of the features (first dimension = number of observations, second dimensions = number of features)
print('\n This is iris.data.shape \n' , iris.data.shape)

#print shape of the response
print('\n This is iris.target.shape \n' , iris.target.shape)

#store feature matrix in 'X'
X = iris.data

#store response vector in 'y'
y = iris.target

#verify the shapes of X and y
print('\n This is X.shape \n' , X.shape)
print('\n This is y.shape \n' , y.shape)

##SCIKIT LEARN 4-STEP MODELING PATTERN
#1/ IMPORT THE CLASS YOU PLAN TO USE
from sklearn.neighbors import KNeighborsClassifier

#2/ INSTANTIATE THE ESTIMATOR (MODEL) #now tune the parameters (name does not matter
#Some people call this knn "clf" (classifier) or "est" (estimator) . If not set will go to default)
knn = KNeighborsClassifier(n_neighbors = 1)
#let's see the default values
print('\n This is knn itself\n', knn)

#3/ FIT THE MODEL WITH DATA ("model training")
knn.fit(X,y)

#4/ PREDICT THE RESPONSE FOR A NEW OBSERVATION (in this example we imagine we have an object with those numbers
#as measurements [3,5,4,2],[5,4,3,2] and we ask the model to predict the response)

#if we have a simple array
# test = knn.predict([[3, 5, 4, 2]])
# print(test)

#if we have a double array we use...
X_new = knn.predict([[3,5,4,2],[5,4,3,2]])
print('\n This is X_new prediction \n' , X_new)

counter = 0
for el in X_new:
    if el == 0:
        print('\nThe element', counter, 'is Setosa ')
    elif el == 1:
        print('\nThe element', counter, 'is Versicolor ')
    elif el == 2:
        print('\nThe element', counter, 'is Virginica ')
    counter =+1


# print('EXPERIMENTAL BELOW THIS LINE .............................')
# #NOW WE CREATE SOME RANDOM NUMPY ARRAYS TO TEST THE MODEL FURTHER
# #https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.randint.html
# import numpy as np
# test = np.random.randint(5, size=(100, 4))
# print(test)
#
# test_predict = knn.predict(test)
# print('\n This is test_predict prediction \n' , test_predict)
#
# counter = 0
# for el in test_predict:
#     if el == 0:
#         print('\nThe element', counter, 'is Setosa ')
#     elif el == 1:
#         print('\nThe element', counter, 'is Versicolor ')
#     elif el == 2:
#         print('\nThe element', counter, 'is Virginica ')
#     counter = counter + 1

##########################################
#NOW WE TRAIN AND TEST THE ENTIRE DATASET
##########################################

from sklearn.linear_model import LogisticRegression

logref = LogisticRegression()
logref.fit(X,y)
logref.predict(X)
y_pred = logref.predict(X)
print('\n I am len(y_pred) \n',len(y_pred))
print('\n I am logref \n', logref)
print('\n I am y_pred \n',y_pred)


#NOW WE PREDICT ACCURACY SCORE CHANGING VALUES OF n_neighbors
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
knn.predict(X)
y_pred = knn.predict(X)
print('\n I am the accuracy score for y and y_pred  \n', metrics.accuracy_score(y,y_pred))

'''''''''''''''''''''''''''''''''''''''''
now we split the data
'''''''''''''''''''''''''''''''''''''''''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state= 4)

print('\n These are the current shapes of X_train, Xtest, y_train and y_test:')
print('\n The value of X_train.shape is:', X_train.shape)
print('\n The value of X_test.shape is:, ', X_test.shape)
print('\n The value of y_train.shape is:', y_train.shape)
print('\n The value of y_test.shape is:', y_test.shape)

#NOW WE COMPARE THE ACCURACY SCORES OF 25 DIFFERENT VALUES FOR K
#TO SEE WHICH ONE GIVES THE BEST ACCURACY
#try K=1 through K=25 and record testing accuracy
#from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

#import matplotlib and plot
import matplotlib.pyplot as plt
#%matplotlib inline if working on console or Jupyter
print('Plot will appear shortly...')
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing accuracy')
plt.show()