#Script to create classification model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, image_to_array, load_img


class classification(object):
	def __init__(self):
		self.entity_model = Sequential()
		self.batch = 20

	def create_classification_model():
		# Layer 1
		self.entity_model.add(Conv2D(64, (3, 3), input_shape=(3, 640, 480)))
		self.entity_model.add(Activation('relu'))
		self.entity_model.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 2
		self.entity_model.add(Conv2D(64, (3, 3)))
		self.entity_model.add(Activation('relu'))
		self.entity_model.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 3
		self.entity_model.add(Conv2D(64, (3, 3)))
		self.entity_model.add(Activation('relu'))
		self.entity_model.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 4(3D feature map converted to 1D feature vector)
		self.entity_model.add(Flatten()) 
		self.entity_model.add(Dense(64))
		self.entity_model.add(Activation('relu'))

		# Layer 5 (Output Layer)
		self.entity_model.add(Dropout(0.5))
		self.entity_model.add(Dense(1))
		self.entity_model.add(Activation('sigmoid'))


		self.entity_model.compile(loss='binary_crossentropy',
					  optimizer='rmsprop',
					  metrics=['accurary'])

		def data_generators(self):
			# Data Generators
			train_datagen = ImageDataGenerator(rescale=1./255)
			test_datagen = ImageDataGenerator(rescale=1./255)

			# Train Data Generator
			train_generator = train_datagen.flow_from_directory(
			        'data/train', 
			        target_size=(150, 150),  
			        batch_size=self.batch,
			        class_mode='binary')

			# Training Validation
			validation_generator = test_datagen.flow_from_directory(
			        'data/validation',
			        target_size=(150, 150),
			        batch_size=self.batch,
			        class_mode='binary')

		def finalise(self):
			self.entity_model.fit_generator(
				train_generator,
			        steps_per_epoch=2000 // self.batch,
			        epochs=50,
			        validation_data=validation_generator,
			        validation_steps=800 // self.batch)

			# Saving the weights 

			self.entity_model.save_weights('weights.h5')
