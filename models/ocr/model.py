#!/usr/bin/env python3
"""
OCR Model for LexoRead

This module provides a model to extract text from images, with special focus
on improving accuracy for dyslexic handwriting and difficult-to-read text.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ocr_model")

class CNNEncoder(nn.Module):
    """
    CNN-based image encoder for OCR.
    
    Uses EfficientNet as backbone for feature extraction.
    """
    
    def __init__(self, pretrained=True):
        """
        Initialize the CNN encoder.
        
        Args:
            pretrained (bool): Whether to use pretrained weights
        """
        super(CNNEncoder, self).__init__()
        
        # Use EfficientNet as backbone
        self.backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
        
        # Remove the classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Feature dimension after backbone
        self.feature_dim = 1280  # EfficientNet-B0 feature dimension
        
    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Encoded features
        """
        # Pass through backbone
        features = self.backbone(x)
        
        # Reshape
        batch_size = features.shape[0]
        features = features.view(batch_size, self.feature_dim, -1)
        features = features.permute(0, 2, 1)  # [batch_size, sequence_length, feature_dim]
        
        return features

class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for OCR.
    
    Takes encoded image features and generates text output.
    """
    
    def __init__(self, vocab_size, hidden_dim=256, nhead=8, num_layers=4, max_length=100):
        """
        Initialize the transformer decoder.
        
        Args:
            vocab_size (int): Size of the vocabulary
            hidden_dim (int): Dimension of the hidden states
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            max_length (int): Maximum sequence length
        """
        super(TransformerDecoder, self).__init__()
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        
        # Feature projection
        self.feature_projection = nn.Linear(1280, hidden_dim)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Parameters
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        
    def forward(self, features, targets=None):
        """
        Forward pass of the decoder.
        
        Args:
            features (torch.Tensor): Encoded image features [batch_size, seq_len, feature_dim]
            targets (torch.Tensor, optional): Target token IDs for teacher forcing
            
        Returns:
            torch.Tensor: Output logits
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Project features to hidden dimension
        features = self.feature_projection(features)
        memory = features.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        
        # Prepare initial input (start token)
        if targets is not None:
            # Teacher forcing
            input_ids = targets[:, :-1]  # Remove last token
            seq_len = input_ids.shape[1]
        else:
            # Inference mode
            input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # Start token
            seq_len = self.max_length
        
        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        tgt = embeddings.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]
        
        # Create attention mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # Decode
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
        
        # Project to vocabulary
        logits = self.output_layer(output)
        
        return logits

class DyslexiaOCRModel(nn.Module):
    """
    Full OCR model combining encoder and decoder.
    """
    
    def __init__(self, vocab_size, hidden_dim=256, pretrained=True):
        """
        Initialize the OCR model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            hidden_dim (int): Dimension of the hidden states
            pretrained (bool): Whether to use pretrained weights
        """
        super(DyslexiaOCRModel, self).__init__()
        
        self.encoder = CNNEncoder(pretrained=pretrained)
        self.decoder = TransformerDecoder(vocab_size, hidden_dim=hidden_dim)
        
    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        
        Args:
            images (torch.Tensor): Input images [batch_size, channels, height, width]
            targets (torch.Tensor, optional): Target token IDs for teacher forcing
            
        Returns:
            torch.Tensor: Output logits
        """
        # Encode images
        features = self.encoder(images)
        
        # Decode to text
        logits = self.decoder(features, targets)
        
        return logits

class DyslexiaOCR:
    """
    Main class for OCR processing.
    
    This class provides methods to load the model, preprocess images,
    and extract text from images.
    """
    
    def __init__(self, model_path=None, device=None, vocab_path=None):
        """
        Initialize the OCR processor.
        
        Args:
            model_path (str, optional): Path to the pre-trained model
            device (str, optional): Device to run the model on ('cuda' or 'cpu')
            vocab_path (str, optional): Path to vocabulary file
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load vocabulary
        self.vocab = self._load_vocabulary(vocab_path)
        
        # Initialize model
        self.model = DyslexiaOCRModel(vocab_size=len(self.vocab))
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model path provided or model not found. Using untrained model.")
            
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Load configurations
        self.config = self._load_config()
        
    def _load_vocabulary(self, vocab_path=None):
        """
        Load vocabulary for the model.
        
        Args:
            vocab_path (str, optional): Path to vocabulary file
            
        Returns:
            dict: Vocabulary (token to ID mapping)
        """
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            return vocab
        else:
            # Default minimal vocabulary
            logger.warning("No vocabulary path provided or file not found. Using minimal vocabulary.")
            vocab = {
                "<pad>": 0,
                "<sos>": 1,
                "<eos>": 2,
                "<unk>": 3
            }
            # Add basic characters
            for i, char in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-'\"()[]{}"):
                vocab[char] = i + 4
            
            return vocab
    
    def _load_config(self):
        """
        Load configuration for the model.
        
        Returns:
            dict: Configuration parameters
        """
        config_path = Path(__file__).parent / 'config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "image_size": 224,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "max_length": 100,
                "beam_size": 5,
                "preprocessing": {
                    "enhance_contrast": True,
                    "denoise": True,
                    "deskew": True
                }
            }
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for OCR.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            image = image_path.copy()
            if image.shape[2] == 3 and image.dtype == np.uint8:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input must be a path or a numpy array")
        
        # Apply preprocessing steps based on config
        if self.config["preprocessing"]["enhance_contrast"]:
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        if self.config["preprocessing"]["denoise"]:
            # Apply denoising
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        if self.config["preprocessing"]["deskew"]:
            # Deskew the image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Find rotation angle
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Resize
        image = cv2.resize(image, (self.config["image_size"], self.config["image_size"]))
        
        # Convert to tensor
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.config["mean"])) / np.array(self.config["std"])
        image = image.transpose(2, 0, 1)  # [C, H, W]
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        return image_tensor
    
    def extract_text(self, image_path, return_confidence=False):
        """
        Extract text from an image.
        
        Args:
            image_path (str): Path to the image
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Extracted text (and confidence scores if requested)
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Extract text
        with torch.no_grad():
            features = self.model.encoder(image)
            
            # Initialize
            batch_size = 1
            input_ids = torch.tensor([[1]], dtype=torch.long, device=self.device)  # Start token
            
            # Decode one token at a time
            output_ids = []
            confidences = []
            
            for i in range(self.config["max_length"]):
                # Forward pass through decoder
                logits = self.model.decoder(features, input_ids)
                
                # Get probabilities for the last token
                probs = F.softmax(logits[0, -1, :], dim=-1)
                
                # Get the token with highest probability
                next_token_id = torch.argmax(probs).item()
                confidence = probs[next_token_id].item()
                
                # Add to outputs
                output_ids.append(next_token_id)
                confidences.append(confidence)
                
                # Stop if end token is generated
                if next_token_id == 2:  # <eos>
                    break
                
                # Update input for next iteration
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)], dim=1)
        
        # Convert token IDs to text
        id_to_token = {v: k for k, v in self.vocab.items()}
        text = "".join([id_to_token.get(token_id, "<unk>") for token_id in output_ids if token_id not in [0, 1, 2]])  # Skip special tokens
        
        if return_confidence:
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            return text, avg_confidence
        else:
            return text
    
    def detect_text_regions(self, image_path):
        """
        Detect regions containing text in an image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            list: List of bounding boxes [(x, y, w, h), ...]
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
        elif isinstance(image_path, np.ndarray):
            image = image_path.copy()
        else:
            raise ValueError("Input must be a path or a numpy array")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small contours
            if w < 10 or h < 10:
                continue
            
            # Filter by aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            bounding_boxes.append((x, y, w, h))
        
        return bounding_boxes
    
    def extract_text_from_regions(self, image_path):
        """
        Extract text from all detected text regions in an image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            list: List of (text, bounding_box) tuples
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_path, np.ndarray):
            image = image_path.copy()
            if image.shape[2] == 3 and image.dtype == np.uint8:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Input must be a path or a numpy array")
        
        # Detect text regions
        bounding_boxes = self.detect_text_regions(image_path)
        
        # Extract text from each region
        results = []
        for box in bounding_boxes:
            x, y, w, h = box
            
            # Extract region
            region = image[y:y+h, x:x+w]
            
            # Extract text from region
            text, confidence = self.extract_text(region, return_confidence=True)
            
            # Add to results if confidence is high enough
            if confidence > 0.5:
                results.append((text, box))
        
        return results

if __name__ == "__main__":
    # Simple demo
    ocr = DyslexiaOCR()
    
    # Example: Extract text from an image
    image_path = "example.jpg"
    if os.path.exists(image_path):
        text = ocr.extract_text(image_path)
        print(f"Extracted text: {text}")
        
        # Detect and extract text from regions
        regions = ocr.extract_text_from_regions(image_path)
        print(f"Found {len(regions)} text regions:")
        for text, box in regions:
            print(f"  - '{text}' at {box}")
    else:
        print(f"Example image not found at {image_path}")
