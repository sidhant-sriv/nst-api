import tensorflow as tf
import numpy as np
import os
# import matplotlib.pyplot as plt


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image, target_size=(224, 224)):
    image = tf.image.resize(image, target_size)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return np.expand_dims(image, axis=0).astype(np.float32)


def deprocess_image(image, original_size):
    x = image.squeeze()
    x = x - np.min(x)
    x = x / np.max(x)
    x = x * 255.0
    x = np.clip(x, 0, 255).astype("uint8")
    x = tf.image.resize(x, original_size)
    return x.numpy()


def run_inference(interpreter, content_image, style_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Set the input tensors
    interpreter.set_tensor(input_details[0]["index"], content_image)
    interpreter.set_tensor(input_details[1]["index"], style_image)
    interpreter.invoke()
    # Get the output tensor
    output_image = interpreter.get_tensor(output_details[0]["index"])
    return output_image


def save_generated_image(
    image_array, filename, output_dir="static", original_size=(224, 224)
):
    os.makedirs(output_dir, exist_ok=True)
    image = deprocess_image(image_array, original_size)
    filepath = os.path.join(output_dir, filename)
    plt.imsave(filepath, image)
    return filepath


def run_style_transfer_tflite(
    content_image_path, style_image_path, tflite_model_path, output_filename
):
    interpreter = load_tflite_model(tflite_model_path)
    # Load and preprocess content image
    content_image = plt.imread(content_image_path)
    original_size = content_image.shape[:2]
    preprocessed_content = preprocess_image(content_image)
    # Load and preprocess style image
    style_image = plt.imread(style_image_path)
    preprocessed_style = preprocess_image(style_image)
    # Run inference
    generated_image = run_inference(
        interpreter, preprocessed_content, preprocessed_style
    )
    # Save and return the output image
    return save_generated_image(
        generated_image, output_filename, original_size=original_size
    )
