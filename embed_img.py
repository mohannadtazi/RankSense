from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

import streamlit as st

multilingue_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")




def embed_image(image, path):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.numpy().flatten(), path





# Function to create image index and store it in JSON
def create_image_index(image_folder, output_json):
    image_index = []  # List to store the embeddings and paths

    for filename in os.listdir(image_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for image files
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)

            # Generate embedding and associate with the image path
            embedding, path = embed_image(image, image_path)
            
            # Store as a dictionary and convert embedding to a list
            image_index.append({
                "embedding": embedding.tolist(),  # Convert NumPy array to list for JSON serialization
                "path": path
            })

    # Save the index to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(image_index, json_file, indent=4)

    print(f"Image index has been created and saved to {output_json}")



# Usage example
image_folder = "images"
output_json = "image_index.json"  # The JSON file to store the index
create_image_index(image_folder, output_json)
