# written by Mohammad Reza Kakoee for CARND Behavioral Clonning Project

import csv
import cv2
import numpy as np
import sklearn

#generator funciton 
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			targets = []
			for line in batch_samples:

				# adding center image and steering meas
				img_filename=line[0].split('/')[-1]
				img_rel_path = "./simulator_train_data/IMG/"+img_filename
				center_image = cv2.imread(img_rel_path)

				#converting image to RGB as cv2 return BGR while drive.py uses RGB	
				center_imgRGB=cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
				images.append(center_imgRGB)
				measurement = float(line[3])
				targets.append(measurement)	
	
				# adding flipped version of center image and -1*steering meas
				image_flipped = np.fliplr(center_imgRGB)
				measurement_flipped = -1*measurement
				images.append(image_flipped)
				targets.append(measurement_flipped)

				#adding left and right images with steering_meas+-delta
				steering_center = measurement
				# create adjusted steering measurements for the side camera images	
				correction = 0.25 # this is a parameter to tune. I found 0.25 good value after a couple of experiments 
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				
				#adding left camera image
				img_left_filename=line[1].split('/')[-1]
				img_left_rel_path = "./simulator_train_data/IMG/"+img_left_filename
				image_left = cv2.imread(img_left_rel_path)
				left_imgRGB=cv2.cvtColor(image_left,cv2.COLOR_BGR2RGB)
				images.append(left_imgRGB)
				targets.append(steering_left)

				#adding right camera image
				img_right_filename=line[2].split('/')[-1]
				img_right_rel_path = "./simulator_train_data/IMG/"+img_right_filename
				image_right = cv2.imread(img_right_rel_path)
				right_imgRGB=cv2.cvtColor(image_right,cv2.COLOR_BGR2RGB)
				images.append(right_imgRGB)
				targets.append(steering_right)

			X_train = np.array(images)
			y_train = np.array(targets)
			yield sklearn.utils.shuffle(X_train, y_train)




lines=[]
csvfile= open("./simulator_train_data/driving_log.csv")
reader=csv.reader(csvfile)
images=[]
targets=[]
BATCH_SIZE=32

for line in reader:
	lines.append(line)

#using 30% for validation 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.3)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Activation,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D,Cropping2D
from keras.layers import Conv2D

# Build Convolutional Neural Network in Keras
model = Sequential()

#preprocessing using Lambda layer - normalize and mean
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160, 320,3)))

#Cropping upper and lower part of image as it does not have road data
model.add(Cropping2D(cropping=((70,25), (0,0))))

#first Conv layer with relu and maxpool
model.add(Convolution2D(24, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D())

#second Conv layer with relu and maxpool
model.add(Convolution2D(36, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D())

#Third Conv layer with relu
model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D())

#Forth/last Conv layer with relu
model.add(Convolution2D(36, 3, 3))
model.add(Activation('relu'))


#first fully connected layer follow up by dropout layer as FC layers tend to overfit
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#second fully connected layer 
model.add(Dense(84))
#third fully connected layer 
model.add(Dense(10))
#last layer to output
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/BATCH_SIZE, validation_data=validation_generator,validation_steps=len(validation_samples)/BATCH_SIZE, nb_epoch=6)


model.save('model.h5')


