

def load(path):
    import tensorflow as tf
    model = tf.keras.models.load_model(path)
    return model