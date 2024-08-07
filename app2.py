import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained CNN model
model = tf.keras.models.load_model('C:/Users/user/Desktop/mywork/Prediscan/dr_model.h5')

def preprocess_image(image_path):
    """
    Preprocess the image to the format required by the CNN model.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease_stage(image_path):
    """
    Predict the disease stage of the retinal image using the CNN model.
    """
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_values = prediction[0]
    return predicted_values

def visualize_image(image_path):
    """
    Display the image to ensure it's loaded and preprocessed correctly.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'C:/Users/user/Desktop/mywork/Prediscan/uploads/extracted_images/Retinal Image Folder/IMG004R.png'

# Visualize the image
# visualize_image(image_path)

# Predict and display stages
predicted_stage = predict_disease_stage(image_path)
print('The predicted disease stage for the given retinal image is ',predicted_stage[0])

