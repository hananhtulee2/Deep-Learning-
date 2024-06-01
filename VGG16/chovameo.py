from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

train_datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

training_set=train_datagen.flow_from_directory('/kaggle/input/cat-and-dog/training_set',
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode='binary')

test_set=train_datagen.flow_from_directory('/kaggle/input/cat-and-dog/test_set',
                                              target_size=(64,64),
                                              batch_size=32,
                                              class_mode='binary')

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation=('relu')))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

cat_image=image.load_img('/kaggle/input/cat-and-dog/test_set/test_set/cats/cat.4003.jpg',target_size=(64,64))


cat_image=image.img_to_array(cat_image)
cat_image = np.expand_dims(cat_image, axis=0) 
result=model.predict(cat_image)
test_set.class_indices
if result[0][0]==1:
    predication='dog'
else:
    predication='cat'
print(predication)