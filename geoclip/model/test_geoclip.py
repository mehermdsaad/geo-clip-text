import torch
from PIL import Image
import os
from transformers import CLIPModel, AutoProcessor

# Import your GeoCLIP class (ensure it's in your PYTHONPATH)
from GeoCLIP.py import GeoCLIP  # Replace 'your_module' with the actual module name

# Set the paths
file_dir = 'path/to/your/files'  # Replace with your actual file directory
image_path = '../images/Kauai.png'  # Replace with your image file path

# Initialize the model
model = GeoCLIP(from_pretrained=True)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Optionally, you can add some descriptive text
text = "A sunny beach with palm trees"

# Predict the location using the image and text
top_k = 5  # Number of top predictions to return
top_pred_gps, top_pred_prob = model.predict(image_path, top_k, text)

# Print the results
print("Top predicted GPS coordinates:")
print(top_pred_gps)
print("Top predicted probabilities:")
print(top_pred_prob)
