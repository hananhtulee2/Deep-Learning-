# Hàn Anh Tú - 21161215
# Nguyễn Minh Nhựt - 21161166
# Final AI Project

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models

import pathlib
training_dir = pathlib.Path('C:/MyLaptop/Spyder File Code/finalproject/training_set')
training_count = len(list(training_dir.glob('*/*.png')))
print(training_count)

test_dir = pathlib.Path('C:/MyLaptop/Spyder File Code/finalproject/test_set')
test_count = len(list(test_dir.glob('*/*.png')))
print(test_count)

batch_size = 64

img_height = 128
img_width = 128
train_ds = tf.keras.utils.image_dataset_from_directory(
  training_dir,
  validation_split=0,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0,
  seed=113,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape),
        layers.Conv2D(8, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),  
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Sử dụng hàm create_model để tạo mô hình
input_shape = (128, 128, 3)
num_classes = len(class_names)  # Cần phải định nghĩa class_names trước khi sử dụng
model = create_model(input_shape, num_classes)
model.summary()

# Chạy Model trên 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Xác đinh learning rate và gọi lại hàm để giảm 
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=3, verbose=1, mode='auto')

# Huấn luyện mô hình
epochs = 40
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs, shuffle=True, callbacks=[learning_rate_reduction])

# Đánh giá mô hình
accuracy = model.evaluate(val_ds)

# Vẽ kết quả 
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

# Chọn ngẫu nhiên một bức ảnh từ tệp test
test_image_paths = list(test_dir.glob('*/*.png'))
random_test_image_path = np.random.choice(test_image_paths)

# Đọc và tiền xử lý bức ảnh
img = keras.preprocessing.image.load_img(
    random_test_image_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Thêm một chiều để phù hợp với batch size

# Dự đoán lớp của bức ảnh
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# In kết quả dự đoán và hiển thị hình ảnh
print("Predicted class:", class_names[predicted_class])
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("Predicted class: {}".format(class_names[predicted_class]))
plt.axis("off")
plt.show()