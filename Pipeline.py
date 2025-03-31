import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam


# Ploting images with landmarks
def plot_image_landmarks(img_array, df_landmarks, index):
    plt.imshow(img_array[index, :, :, 0], cmap = 'gray')
    plt.scatter(df_landmarks.iloc[index][0: -1: 2], df_landmarks.iloc[index][1: : 2], c = 'y')
    plt.show()

def plot_img_preds(images, truth, pred, index):
    plt.imshow(images[index, :, :, 0], cmap='gray')

    t = np.array(truth)[index]
    plt.scatter(t[0::2], t[1::2], c='y')

    p = pred[index, :]
    plt.scatter(p[0::2], p[1::2], c='r')

    plt.show()

features = np.load('dataset/face_images.npz')
features = features.get(features.files[0]) # images
features = np.moveaxis(features, -1, 0)
features = features.reshape(features.shape[0], features.shape[1], features.shape[1], 1)

keypoints = pd.read_csv('dataset/facial_keypoints.csv')
keypoints.head()

# Cleaing data
keypoints = keypoints.fillna(0)
num_missing_keypoints = keypoints.isnull().sum(axis = 1)
print(num_missing_keypoints)

new_features = features[keypoints.index.values, :, :, :] #Nums of rows,w, H, Channels
new_features = new_features / 255
keypoints.reset_index(inplace = True, drop = True)

plot_image_landmarks(new_features, keypoints, 18)

x_train, x_test, y_train, y_test = train_test_split(new_features, keypoints, test_size=0.2)

img_size = 96

model = Sequential()

model.add(Input(shape=(img_size, img_size, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding="same",kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding="same",kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding="same",kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256,kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0.1))

model.add(Dropout(0.5))

model.add(Dense(64,kernel_initializer=glorot_uniform()))
model.add(LeakyReLU(alpha=0))

model.add(Dense(30,kernel_initializer=glorot_uniform()))

model.summary()
model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

BATCH_SIZE = 200
EPOCHS = 2

history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test),
    shuffle=True,
    verbose=1,
)

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Accuracy for Facial keypoints')
plt.ylabel('Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# plt.plot(history.history['mean_squared_error'], label='MSE (training data)')
# plt.plot(history.history['val_mean_squared_error'], label='MSE (validation data)')
plt.plot(history.history['loss'], label='MSE (training data)')
plt.plot(history.history['val_loss'], label='MSE (validation data)')
plt.title('MSE for Facial keypoints')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

y_pred = model.predict(x_test)
print(y_pred)


plot_img_preds(x_test, y_test, y_pred, 3)
plot_img_preds(x_test, y_test, y_pred, 18)
