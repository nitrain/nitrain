def load_model(path):
    import tensorflow as tf
    model_path = f'{path}.keras'
    model = tf.keras.models.load_model(model_path)
    return model