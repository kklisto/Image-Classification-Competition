import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from keras import backend as K
import cv2

np.random.seed(42)
tf.random.set_seed(42)

class Model:
    def __init__(self):
        # Define h-swish activation function, only for (final) MobileNetV3 submissions
        def h_swish(x):
            return x * tf.nn.relu6(x + 3.0) / 6.0

        self.neural_network = tfk.models.load_model('<filename>.keras', custom_objects={'h_swish': h_swish})

    def predict(self, X):

        # Define class weights (e.g., higher for hard classes), only for custom loss submission
        # class_weights = [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.5, 1.0]
        # loss_fn = WeightedCategoricalCrossentropy(class_weights=class_weights)

        normalized_X = (X / 127.5).astype('float32') - 1  # Normalize to [-1, 1]
        #resized_X = np.array([  # Only needed for pre-trained MobileNetV2 submission
        #    cv2.resize(img, (224, 224),
        #               interpolation=cv2.INTER_CUBIC) for img in normalized_X])
        
        preds = self.neural_network.predict(normalized_X, verbose=0)
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)

        return preds
    
    
@tfk.saving.register_keras_serializable()  # Only needed for custom loss submission
class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, name="weighted_categorical_crossentropy"):
        """
        Custom loss function for weighted categorical crossentropy.
        
        Args:
        - class_weights (list or tensor, optional): A list or tensor of weights for each class.
        - name (str, optional): Name of the loss function.
        """
        super(WeightedCategoricalCrossentropy, self).__init__(name=name)
        
        if class_weights is not None:
            self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        else:
            self.class_weights = None

    def call(self, y_true, y_pred):
        """
        Computes the weighted categorical crossentropy between true labels and predictions.
        
        Args:
        - y_true: Ground truth labels.
        - y_pred: Predicted probabilities.
        
        Returns:
        - loss: Weighted categorical crossentropy loss value.
        """
        # Clip predictions to prevent log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Compute the categorical crossentropy
        crossentropy = -K.sum(y_true * K.log(y_pred), axis=-1)
        
        if self.class_weights is not None:
            # Apply the class weights
            weights = K.sum(self.class_weights * y_true, axis=-1)
            loss = crossentropy * weights
        else:
            loss = crossentropy
        
        return K.mean(loss)