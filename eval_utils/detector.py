import torch
import numpy as np
from typing import List, Dict, Any
from PIL import Image

class ReferringDetector:
    def __init__(self, model, device="cuda"):
        """
        Initializes the detector with a visual grounding model.
        
        Args:
            model: A pre-trained visual grounding model (e.g., a Transformer-based model).
            device: The computation device ("cuda" or "cpu").
        """
        self.model = model.to(device).eval()
        self.device = device
        self.image_cache = None  # Stores processed image to avoid redundant computation
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocesses the image into a format suitable for the model.
        
        Args:
            image: A PIL Image object.

        Returns:
            A processed image tensor.
        """
        # Example transformation (modify as per model requirements)
        image_tensor = torch.tensor(np.array(image)).float() / 255.0  # Normalize
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (B, C, H, W)
        return image_tensor

    def run_detector(self, image: Image.Image, expressions: List[str]) -> Dict[str, Any]:
        """
        Runs the detector on an image with multiple referring expressions.
        
        Args:
            image: A PIL Image.
            expressions: A list of referring expressions.

        Returns:
            A dictionary mapping expressions to detected bounding boxes or masks.
        """
        if self.image_cache is None or self.image_cache["image"] != image:
            self.image_cache = {
                "image": image,
                "processed": self.preprocess_image(image)
            }
        
        processed_image = self.image_cache["processed"]
        results = {}

        for expression in expressions:
            # Run model inference (modify based on model specifics)
            with torch.no_grad():
                output = self.model(processed_image, expression)  # Assuming model takes (image, text)
            
            results[expression] = output  # Store results (e.g., bounding box or segmentation mask)

        return results

# Example Usage
# Assuming `visual_grounding_model` is a loaded model instance with a callable forward method
# detector = ReferringDetector(model=visual_grounding_model)
# image = Image.open("sample.jpg")
# results = detector.run_detector(image, ["a red car", "a black cat"])
# print(results)
