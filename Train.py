#Ai with python
from keras.models import Sequential
#its initialize the neural network as Sequential nw
from keras.layers import Conv2D
# it is used to convolution operation ,it is the 1st process of CNN
from keras.layers import MaxPooling2D
#its is used to perform the poling operation ,it is 2nd process of CNN
from keras.layers import Flatten
#it is used to process of converting all the resultant 2D arrays into a single long continuous linear vector
from keras.layers import Dense,Activation
#it is used to perform the full connection of the neural network
import warnings
warnings.filterwarnings(&#39;ignore&#39;)

#start coding
model=Sequential() #Now create an Object of Sequential class
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation=&#39;relu&#39;)) #nxt step Convolution part
#relu is the rectifier fun
model.add(MaxPooling2D(pool_size=(2,2))) #after convolution part pooling operation will
convert the features map

model.add(Conv2D(32,3,3))
model.add(Activation(&#39;relu&#39;))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,3,3))

model.add(Activation(&#39;relu&#39;))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #convert continuous vector
model.add(Dense(units=2,activation=&#39;relu&#39;)) #create a fully connected to layer#here 128 is
the no. of hidden units. it is a common practice to define the no. of hidden units as the power of 2.
#model.add(Dense(4))
#model.add(Dense(3))
#it initialize the output layer
model.add(Activation(&#39;softmax&#39;))
model.summary() #modelsummary
model.compile(optimizer=&#39;adam&#39;,loss=&#39;categorical_crossentropy&#39;,metrics=[&#39;accuracy&#39;]) # it compile the
CNN

train= train.csv

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=T
rue)
#it perform the image agumentation and then fit the image to neural network
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(train,target_size=(64,64),batch_size=64,class_mode=&#39;c
ategorical&#39;)
test_set=test_datagen.flow_from_directory(validation,target_size=(64,64),batch_size=64,class_mode=&#39;c
ategorical&#39;)
#history = model.fit_generator( train_generator, callbacks = callbacks,
samples_per_epoch=nb_train_samples,

#nb_epoch=nb_epochs, validation_data=validation_generator,
nb_val_samples=nb_val_samples)

#model.fit_generator(training_set,steps_per_epoch=5,epochs=5,validation_data=test_set,validation_st
eps=2)
#it fit the data to the model we have created
#here steps_per_epoch have the no.of training images
# Training with callbacks
from keras import callbacks
&#39;&#39;&#39;
filename=&#39;Train&#39;

&#39;&#39;&#39;
early_stopping=callbacks.EarlyStopping(monitor=&#39;val_loss&#39;, min_delta=0, patience=0, verbose=0,
mode=&#39;min&#39;)

filepath=&quot;Best-weights-my_model.hdf5&quot;

checkpoint = callbacks.ModelCheckpoint(filepath, monitor=&#39;val_loss&#39;, verbose=1, save_best_only=True,
mode=&#39;min&#39;)

callbacks_list = [early_stopping,checkpoint]

#hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1,
validation_data=(X_test, y_test),callbacks=callbacks_list)
#model.fit_generator(training_set,steps_per_epoch=5,epochs=5,validation_data=test_set,validation_st
eps=2,callbacks=callbacks_list)
history = model.fit_generator( training_set, callbacks = callbacks_list, steps_per_epoch=5,

epochs=10, validation_data=test_set,validation_steps=5)
print(&quot;Training Completed&quot;)
