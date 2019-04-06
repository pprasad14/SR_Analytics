
## Import numpy and keras library with features like convolution,max pooling,fully connected(dense) etc.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import backend as K
import os
import numpy as np

# dimensions of our images.
img_width, img_height = 100, 100
# setting training data and test(validation) data directory and providing its information like sample size ,epochs 
train_data_dir = 'D:/ImageAnalytics/diaretdb1_v_1_/resources/retina(classified)/Train'
validation_data_dir = 'D:/ImageAnalytics/diaretdb1_v_1_/resources/retina(classified)/Test'
# describing total sample size of training and validation(test) set
nb_train_samples = 982
nb_validation_samples = 330
# number of epochs
epochs = 50
batch_size = 16
# shrink  every image to pixel size 100 x 100 x 3
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# start the CNN Modelling
model = Sequential()
# layers of the Neural Network
model.add(Conv2D(32, (3, 3), input_shape=input_shape))      # Convolution layer with 32x32x3 pixel size
model.add(Activation('relu'))                               # Rectified Linear Unit layer
model.add(MaxPooling2D(pool_size=(2, 2)))                   # Max pooling  layer

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Preparing the layer for dropout( generalization of model)
# 3 D to 1 D pixel size >> Flattening
model.add(Flatten())
#Fully connected layer
model.add(Dense(64))
model.add(Activation('relu'))
#dropout to reduce overfitting
model.add(Dropout(0.5))
model.add(Dense(1))
# convert the data to probability for each class
model.add(Activation('sigmoid'))
# configuring the learning process
model.compile(loss='binary_crossentropy',  # binary crossentropy for 2 classes ( multiple for many)
              optimizer='rmsprop',             #optimize using gradient descent 
              metrics=['accuracy'])             # check for accuracy

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# training set 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
#test set
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#fine tuning the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
# saving the weights decided by this CNN model for future use 
model.save_weights('diabetic_retino.h5')


# Created a function which will :
    # 1) resize the image to 100x100 pixel matrix  
    #2) predict the result ( diabetic or normal) using  the CNN Model prev generated
def classifier(file):
    test_image = image.load_img(file, target_size = (100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    train_generator.class_indices
    #if result = 0 then it is non diabetic, if result =1 then it is diabetic
    if result[0][0] == 0:
        prediction = 'Non Diabetic'
    else:
        prediction = 'Diabetic'
    return prediction

# applying the function to a single retina scan image file
print(classifier('D:/ImageAnalytics/Valid/image015.jpg'))

# applying the function to a folder full of retina scanned images with ListImg containing the image directory
listImg=os.listdir('D:/ImageAnalytics/Valid/')
# for all the images in the target image directory, print their names as well as the prediction
for t in listImg:
    print(t)
    print(classifier('D:/ImageAnalytics/Valid/'+t))
