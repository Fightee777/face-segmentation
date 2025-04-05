import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper Functions
def plot_image_landmarks(img_array, df_landmarks, index, scale=96.0):
    """Plot one image with ground-truth facial keypoints."""
    plt.imshow(img_array[index, :, :, 0], cmap='gray')
    keypoints = df_landmarks.iloc[index].values * scale  # scale to pixel coords
    xs = keypoints[0::2]
    ys = keypoints[1::2]
    plt.scatter(xs, ys, c='y')
    plt.title("Ground Truth Keypoints")
    plt.show()

def plot_img_preds_grid(images, truth_df, preds, start_index=0, scale=96.0, rows=5, cols=5):
    """
    Plot a grid (rows x cols) of images, each with ground-truth and predicted keypoints.
    By default, it shows 25 images (5x5) starting from 'start_index'.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    for i in range(rows * cols):
        idx = start_index + i
        ax = axes[i // cols, i % cols]
        ax.imshow(images[idx, :, :, 0], cmap='gray')
        true_pts = truth_df.iloc[idx].values * scale
        pred_pts = preds[idx] * scale
        ax.scatter(true_pts[0::2], true_pts[1::2], c='y', label='Truth')
        ax.scatter(pred_pts[0::2], pred_pts[1::2], c='r', label='Pred')
        ax.set_title(f"Index {idx}")
        ax.axis('off')

    # Create a single legend for the entire figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

# Load .npz file and reshape
features_npz = np.load('dataset/face_images.npz')
features = features_npz[features_npz.files[0]]
print("Initial shape (from .npz):", features.shape)

if features.shape[2] > 1000:
    features = np.moveaxis(features, 2, 0)
    print("Shape after moveaxis:", features.shape)

if features.ndim == 3:
    features = features[..., np.newaxis]
print("Final features shape:", features.shape)

# Load Keypoints & filter
keypoints = pd.read_csv('dataset/facial_keypoints.csv')
print("Original keypoints shape:", keypoints.shape)
keypoints_clean = keypoints.dropna()
print("After dropping NaN:", keypoints_clean.shape)

clean_indices = keypoints_clean.index
features_clean = features[clean_indices, :, :, :] / 255.0  # normalize images
keypoints_clean = keypoints_clean / 96.0  # normalize keypoints
keypoints_clean.reset_index(drop=True, inplace=True)

# Quick visual check for first example
plot_image_landmarks(features_clean, keypoints_clean, index=0)

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    features_clean, keypoints_clean, test_size=0.2, random_state=42
)
print("Train shape:", x_train.shape, y_train.shape)
print("Test shape:", x_test.shape, y_test.shape)

# configurating NN
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(x_train, y_train, batch_size=64, shuffle=True)
test_generator = test_datagen.flow(x_test, y_test, batch_size=64, shuffle=False)

# Build CNN model
model = Sequential([
    Input(shape=(96, 96, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(30)  # 15 keypoints * 2(x, y)
])
model.compile(
    optimizer=Adam(1e-3),
    loss='mean_squared_error',
    metrics=[tf.keras.metrics.MeanSquaredError(name='mse')]
)
model.summary()

# Train the model
EPOCHS = 50
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    verbose=1
)

# Plot training curves
plt.plot(history.history['mse'], label='MSE (train)')
plt.plot(history.history['val_mse'], label='MSE (val)')
plt.title('Training vs Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Loss (train)')
plt.plot(history.history['val_loss'], label='Loss (val)')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate predictions
y_pred = model.predict(x_test)

# Final NxN grid with predicted vs truth keypoints
plot_img_preds_grid(x_test, y_test, y_pred, start_index=0, scale=96.0, rows=3, cols=3)
