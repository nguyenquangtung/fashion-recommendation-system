import tensorflow as tf
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
class FashionRecommendationModel:
    def __init__(self):
        # Define and build the model in the constructor
        self.model = self.build_model()

    def build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            GlobalMaxPooling2D()
        ])

        return model