# Here we define the augmentation pipeline, custom augmentation layers,
# and the functions to apply the pipeline to the dataset and visualize the augmented images.
import tensorflow as tf
from keras_cv import layers as kcvl
from tensorflow.keras.layers import Resizing # type: ignore
import matplotlib.pyplot as plt

# Augmentation pipeline parameters
value_range = (0, 255)  # Range of pixel values in the images

# Define the augmentation layers
enhancement_layers = [
    kcvl.RandAugment(value_range=value_range, magnitude=0.2, magnitude_stddev=0.2),
    kcvl.GridMask(ratio_factor=0.4),
    kcvl.RandomSharpness(factor=0.2, value_range=value_range)
]

# Custom function to add Gaussian noise
def add_gaussian_noise(images, mean=0.0, stddev=20):
    noise = tf.random.normal(shape=tf.shape(images), mean=mean, stddev=stddev, dtype=tf.float32)
    noisy_images = images + noise
    return tf.clip_by_value(noisy_images, 0.0, 255.0)

# Define the augmentation pipeline
pipeline = kcvl.RandomAugmentationPipeline(
    layers=enhancement_layers,
    augmentations_per_image=4
)

def apply_pipeline(inputs):
    inputs["images"] = pipeline(inputs["images"])  # Apply RandomAugmentationPipeline
    inputs["images"] = add_gaussian_noise(inputs["images"])  # Add custom Gaussian Noise
    return inputs

# Function to plot augmented images for visualization
def plot_augmented_images(X_train, y_train, num_images, num_augmentations=4):
    batch = X_train[:num_images]  # Take a batch of images for visualization
    labels_batch = y_train[:num_images]

    fig, axes = plt.subplots(num_augmentations + 1, num_images, figsize=(20, 10))  # 4 rows, 10 columns

    # Display original images in the first row
    for i in range(num_images):
        original_image = batch[i].astype("uint8")  # Convert to uint8 for display
        axes[0, i].imshow(original_image)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12)
        else:
            axes[0, i].set_title("")

    # Display augmentations in subsequent rows
    for j in range(num_augmentations):
        augmented_batch = apply_pipeline({
            "images": tf.convert_to_tensor(batch),
            "labels": tf.convert_to_tensor(labels_batch, dtype=tf.float32)
        })
        for i in range(num_images):
            augmented_image = tf.clip_by_value(augmented_batch["images"][i], 0, 255)  # Ensure values are in display range
            axes[j + 1, i].imshow(augmented_image.numpy().astype("uint8"))
            axes[j + 1, i].axis("off")
            if i == 0:
                axes[j + 1, i].set_title(f"Augmentation {j + 1}", fontsize=12)
            else:
                axes[j + 1, i].set_title("")

    plt.tight_layout()
    plt.show()

# Function to create the training dataset pipeline with augmentation and normalization
def create_train_dataset(X_train, y_train, batch_size, target_size):
    normalization = 127.5
    range = 1
    resize_layer = Resizing(target_size[0], target_size[1], interpolation='bicubic')

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .batch(batch_size)
        .map(lambda x, y: (apply_pipeline({"images": x, "labels": tf.cast(y, tf.float32)})["images"], y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (resize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)  # Apply bicubic resizing
        .map(lambda x, y: ((x / normalization) - range, y), num_parallel_calls=tf.data.AUTOTUNE)  # Normalize images
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_dataset

# Function to create the validation dataset pipeline with resizing and normalization
def create_val_dataset(X_val, y_val, batch_size, target_size):
    normalization = 127.5
    range = 1
    resize_layer = Resizing(target_size[0], target_size[1], interpolation='bicubic')

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .map(lambda x, y: (resize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)  # Apply bicubic resizing
        .map(lambda x, y: ((x / normalization) - range, y), num_parallel_calls=tf.data.AUTOTUNE)  # Normalize images
        .prefetch(tf.data.AUTOTUNE)
    )
    return val_dataset
