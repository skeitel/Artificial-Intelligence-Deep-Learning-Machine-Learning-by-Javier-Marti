#Loading our own data into a deep learning framework using OS and pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = 'C:/Users/nique/PycharmProjects/untitled/PetImages/PetImages'
CATEGORIES = ['Dog','Cat']

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #path to cats or dogs directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        #checking first picture
        plt.imshow(img_array, cmap = 'gray')
        plt.show()
        break
    break

#checking type of array we have
#print(img_array.shape)


#checking the image at different sizes to see which size is the minimum where the features are still readable
# IMG_SIZE = 50
# new_array  = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap = 'gray')
# plt.show()

IMG_SIZE = 50
training_data = []
def create_training_data():
    #IMG_SIZE = 50
    for category in CATEGORIES:
        try:
            path = os.path.join(DATADIR, category)  # path to cats or dogs directory
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
        except Exception as e:
            pass

create_training_data()

#Quality check len of training data
print('Len of training data is:', len(training_data))

#Shuffling the data
import random
random.shuffle(training_data)

#QCheck an example of training data
# for sample in training_data[:10]:
#     print(sample[1])


#Create lists of features and labelsa
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

#reshape data
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #the "1" is because it's in grayscale

#THis is all is required to load data into the model. Below is how to save the data for latter use
'''
import pickle
#saving the data for X
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

#saving the data for y
pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)

#QC print
print(X[1])

'''