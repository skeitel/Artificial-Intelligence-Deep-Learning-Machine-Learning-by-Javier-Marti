#Modern implementation of deep learning using Tensorflow and Keras
import tensorflow as tf
import matplotlib.pyplot as plt

#we load the data
mnist = tf.keras.datasets.mnist

#we assign the variables to the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#we normalize the data so it falls in values between 0 and 1 to speed up computation and save in memory load
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#we build the model
model = tf.keras.models.Sequential()

#now we set up our layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#choose parameters for the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#now we fit the model. From here we are ready to run it
model.fit(x_train, y_train, epochs = 3)

#Now we print the final calculation loss and calculation accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)

#we print our results. It's normal that our final results will be slightly different from the ones gotten whilst the model was running. However, if the val_loss and the val_acc differ too much from the previous results, we must tweak the model, as this may be prone to overfitting.
print('..............\n')
print('val_loss (loss) is:')
print(val_loss)
print('val_acc (accuracy) is:')
print(val_acc)

#To save the model we use:
#model.save('name_we_give_to_the_model'.model)

#To LOAD a model we use:
#new_model = tf.keras.models.load_model('name_we_give_to_the_model')

#To make a prediction (NOTE THAT PREDICTION ALWAYS TAKES A LIST)
#predictions = new_model.predict([x_test]) #note the square brackets to signify it's a list

#Now testing by doing the prediction for the first value in the list of values passed to it
#import numpy as np
#print(np.argmax(predictions[0]) #this will give us the value of the prediction for the first item (item 0) in the list we are passing it (x_test)
#Now if we want to check the accuracy of the prediction wee can print the image corresponding to that item (item 0) by typing:
#plt.imshow(x_test[0])
#plt.show()
