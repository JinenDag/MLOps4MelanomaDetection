#Step 1: Import the required packages
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt
# % matplotlib inline
# Step 2: ImageDataGenerator: data augmentation

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

#  Load the training Set and test set
training_set = train_datagen.flow_from_directory('base_melanome_CNN/training_set',
                                                 target_size = (124, 124),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('base_melanome_CNN/test_set',
                                                 target_size = (124, 124),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')
num_classes = training_set.num_classes
class_names = list(training_set.class_indices.keys())
num_classes , class_names


# Step 3: Initialising the CNN

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (124, 124, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.4))


model.add(Flatten())# this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()
# Step 4: Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history= model.fit_generator(training_set,
                          epochs = 5,
                          validation_data = test_set)
# list all data in history
print(history.history.keys())
#step5: Show the resulats

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

result = model.evaluate_generator(test_set, steps=1)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))
result = model.evaluate_generator(training_set, steps=1)
print("Train-set classification accuracy: {0:.2%}".format(result[0]))


# step 6: Prediction

test_image = load_img('base_melanome_CNN/cas.jpg', target_size = (124,124))
plt.imshow(test_image, interpolation = 'spline16')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

test_image = np.expand_dims(test_image, axis = 0)
result= model.predict(test_image)
t=0
i=0
for label in class_names:
        print("\t%s ==> %.2f %%" % (label, result[t][i]*100))
        i = i + 1