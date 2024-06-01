import numpy as np
import matplotlib.pyplot as plt  
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the model
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Flatten(),
    layers.Dense(4, activation="relu"),  # Hidden layer with ReLU activation
    layers.Dense(4, activation="sigmoid"),  # Hidden layer with Sigmoid activation
    layers.Dense(num_classes, activation="softmax")  # Output layer with Softmax activation
])

model.summary()

# Set the optimizer and batch size
optimizer = keras.optimizers.Adam()
batch_size = 200
epochs = 5

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the model using mini-batch gradient descent
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
#print(x_train[0])

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Select an image for prediction
image_index = 0 
test_image = x_test[image_index]
test_image = np.expand_dims(test_image, axis=0)

# Make prediction
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions, axis=1)
print(f"Class predicted: {predicted_class}")

# Display the image and predicted class
plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted class: {predicted_class}")
plt.show()
