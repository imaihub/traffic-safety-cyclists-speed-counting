import torch
import cv2
import numpy as np
from PIL import Image

class DETRFeatureExtractor:
    """
    A class to extract features from image patches using a pre-trained DETR model.
    This is typically used for generating embeddings for tracking algorithms like DeepSort.
    """
    def __init__(self, model, processor, device):
        """
        Initializes the feature extractor.

        Args:
            model (Detr): An instance of the DETR model (from elements/models.py).
            processor (DetrImageProcessor): The image processor from Hugging Face Transformers.
            device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the model on.
        """
        self.model = model
        self.processor = processor
        self.device = device

    def __call__(self, image_patch):
        """
        Extracts features from a given image patch.

        Args:
            image_patch (numpy.ndarray): The input image patch (e.g., a bounding box crop).
                                         Expected format: HxWxC (RGB or BGR).

        Returns:
            numpy.ndarray: A 1D numpy array representing the extracted features.
        """
        # Convert BGR to RGB if the input image patch is in BGR format (common for OpenCV)
        if image_patch.shape[2] == 3 and image_patch.dtype == np.uint8: # Check for 3 channels and byte type
            # Assuming OpenCV reads images as BGR, convert to RGB for PIL/Transformers
            image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array image patch to a PIL Image object, as required by the processor.
        patch_pil = Image.fromarray(image_patch)

        # Preprocess the image patch using the DETR image processor.
        # `return_tensors="pt"` ensures PyTorch tensors are returned.
        # Move the inputs to the specified device (CPU/GPU).
        inputs = self.processor(images=patch_pil, return_tensors="pt").to(self.device)

        # Perform inference without gradient computation to save memory and speed up.
        with torch.no_grad():
            # Pass the processed inputs to the DETR model.
            outputs = self.model(**inputs)
            # Extract features from the last hidden state of the model.
            # `mean(dim=1)` averages features across spatial dimensions to get a single vector per patch.
            # Move to CPU and convert to NumPy array.
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Return the features for the first (and likely only) patch in the batch as a 1D array.
        return features[0]
