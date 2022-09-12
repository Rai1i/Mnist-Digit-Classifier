import numpy as np 
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Input,Dropout,concatenate,BatchNormalization
from keras.models import Model
from sklearn import tree
from os.path import dirname, abspath, join
import matplotlib.pyplot as plt

# Classic get file locations stuff 

project_dir = dirname(abspath(__file__))
test_data_dir = join(join(project_dir, 'archive'), 'test.csv')


# Load the puppy in via pandas, thank Mr.Kaggle for making the data nice for us

(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
test = pd.read_csv(test_data_dir)
test = test.values.reshape(28000,28,28,1) 

# Making our data pretty, the reshaping of X restructures to work for a CNN

image_size = x_train.shape[1]
input_shape = (image_size,image_size,1)


def create_model():
    left_inputs = Input(shape = input_shape)
    x = left_inputs
    x = Conv2D(filters = 64,kernel_size = 2,padding = 'same',strides = 1,activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.27)(x)

    x = Conv2D(filters = 64,kernel_size = 2,padding = 'same',strides = 1,activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.27)(x)

    x = Conv2D(filters = 64,kernel_size = 2,padding = 'same',strides = 1,activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters = 128,kernel_size = 2,padding = 'same',strides = 1,activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.27)(x)

    x = Conv2D(filters = 256,kernel_size = 2,padding = 'same',strides = 1,activation = 'relu')(x)

    right_inputs = Input(shape = input_shape)
    y = right_inputs
    y = Conv2D(filters = 64,kernel_size = 2,padding = 'same',strides = 1,dilation_rate = 3,activation = 'relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D()(y)
    y = Dropout(0.27)(y)

    y = Conv2D(filters = 64,kernel_size = 2,padding = 'same',strides = 1,dilation_rate = 3,activation = 'relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D()(y)
    y = Dropout(0.27)(y)

    y = Conv2D(filters = 64,kernel_size = 2,padding = 'same',strides = 1,dilation_rate = 3,activation = 'relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D()(y)
    y = Dropout(0.3)(y)

    y = Conv2D(filters = 128,kernel_size = 2,padding = 'same',strides = 1,dilation_rate = 3,activation = 'relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D()(y)
    y = Dropout(0.27)(y)
    y = Conv2D(filters = 256,kernel_size = 2,padding = 'same',strides = 1,activation = 'relu')(y)

    y = concatenate([x,y])
    y = Flatten()(y)
    y = Dropout(0.27)(y)
    y = Dense(20,activation ='relu')(y)
    outputs = Dense(10,activation = 'sigmoid')(y)
    model = Model([left_inputs,right_inputs],outputs)
    tf.keras.utils.plot_model(model)
    return model

model = create_model()
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

history = model.fit([x_train,x_train],y_train,batch_size = 55,epochs = 10, validation_data = ([x_val,x_val],y_val))

model.evaluate([x_val,x_val],y_val)

results = model.predict([test,test])
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("MNIST_04.csv",index=False)
