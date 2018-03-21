from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

(train_features,train_labels),(test_features,test_labels) = cifar10.load_data()

#train features should have value between 0 and 1
train_features = train_features.astype('float32')/255
#
test_features = test_features.astype('float32')/255
#
# #Get uniques classes from the labels. Convert train labels to unique classes
num_classes = len(np.unique(train_labels))
#
train_labels = np_utils.to_categorical(train_labels,num_classes)
#
test_labels = np_utils.to_categorical(test_labels, num_classes)
#
model = Sequential()

model.add(Convolution2D(48,(3,3),border_mode='SAME',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Convolution2D(48,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(96,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(192,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Dense(num_classes,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model

model_info = model.fit(train_features, train_labels,
                       batch_size=128, epochs=200,
                       validation_data = (test_features, test_labels),
                       verbose=0)

print "Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2,horizontal_flip=True)

model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 200,
                                 validation_data = (test_features, test_labels), verbose=0)




