import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
svm_model = joblib.load("svm_model.pkl")

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_flattened = image_array.flatten()
    image_normalized = image_flattened / 255.0
    
    return image_normalized

def predict_digit(image_path):
    image_processed = preprocess_image(image_path)
    image_processed = image_processed.reshape(1, -1)
    prediction = svm_model.predict(image_processed)
    
    return prediction[0]

image_path = "image-examples\hw_5.png"
predicted_digit = predict_digit(image_path)
print("Predicted Digit:", predicted_digit)

# Display the image for visualization
image = Image.open(image_path)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
