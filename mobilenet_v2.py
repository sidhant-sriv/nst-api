import tensorflow as tf


def convert_mobilenet_to_tflite():
    # Load MobileNetV2 model pre-trained on ImageNet
    model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open("mobilenet_v2.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model converted and saved as 'mobilenet_v2.tflite'")


if __name__ == "__main__":
    convert_mobilenet_to_tflite()
