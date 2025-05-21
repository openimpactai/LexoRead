#!/usr/bin/env python3
"""
Export LexoRead models to different formats.

This script exports models from PyTorch format to optimized formats such as ONNX
or TorchScript for deployment in different environments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import torch
import time

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

# Import models
from models.text_adaptation.model import DyslexiaTextAdapterModel, DyslexiaTextAdapter
from models.ocr.model import DyslexiaOCRModel, DyslexiaOCR
from models.tts.model import DyslexiaTTSModel, DyslexiaTTS
from models.reading_level.model import BERTReadingLevelModel, ReadingLevelAssessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("export_model")

# Supported model types
MODEL_TYPES = {
    "text_adaptation": {
        "model_class": DyslexiaTextAdapterModel,
        "wrapper_class": DyslexiaTextAdapter,
        "default_input": {
            "input_ids": torch.randint(0, 30522, (1, 128)),
            "attention_mask": torch.ones(1, 128),
            "token_type_ids": torch.zeros(1, 128)
        }
    },
    "ocr": {
        "model_class": DyslexiaOCRModel,
        "wrapper_class": DyslexiaOCR,
        "default_input": {
            "images": torch.randn(1, 3, 224, 224)
        }
    },
    "tts": {
        "model_class": DyslexiaTTSModel,
        "wrapper_class": DyslexiaTTS,
        "default_input": {
            "text": torch.randint(0, 100, (1, 100))
        }
    },
    "reading_level": {
        "model_class": BERTReadingLevelModel,
        "wrapper_class": ReadingLevelAssessor,
        "default_input": {
            "input_ids": torch.randint(0, 30522, (1, 128)),
            "attention_mask": torch.ones(1, 128),
            "token_type_ids": torch.zeros(1, 128)
        }
    }
}

# Supported export formats
EXPORT_FORMATS = ["onnx", "torchscript"]

def load_model(model_type, model_path=None):
    """
    Load a model from a checkpoint.
    
    Args:
        model_type (str): Type of model to load
        model_path (str, optional): Path to model checkpoint
        
    Returns:
        torch.nn.Module: Loaded model
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Get model class
    model_class = MODEL_TYPES[model_type]["model_class"]
    
    # Create model instance
    if model_type == "text_adaptation":
        model = model_class(bert_model_name="bert-base-uncased", num_labels=3)
    elif model_type == "ocr":
        model = model_class(vocab_size=100)
    elif model_type == "tts":
        model = model_class(vocab_size=100, mel_dim=80)
    elif model_type == "reading_level":
        model = model_class(bert_model_name="bert-base-uncased", num_classes=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights if provided
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        logger.warning(f"No weights loaded. Using randomly initialized model.")
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def export_to_onnx(model, model_type, output_path, opset_version=11):
    """
    Export a model to ONNX format.
    
    Args:
        model (torch.nn.Module): Model to export
        model_type (str): Type of model
        output_path (Path): Path to save the ONNX model
        opset_version (int): ONNX opset version
        
    Returns:
        bool: Whether the export was successful
    """
    try:
        # Get default inputs for the model
        default_input = MODEL_TYPES[model_type]["default_input"]
        
        # Export model to ONNX
        torch.onnx.export(
            model,
            tuple(default_input.values()),
            output_path,
            input_names=list(default_input.keys()),
            output_names=["output"],
            dynamic_axes={name: {0: "batch_size"} for name in default_input.keys()},
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        logger.info(f"Exported model to ONNX: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        return False

def export_to_torchscript(model, output_path):
    """
    Export a model to TorchScript format.
    
    Args:
        model (torch.nn.Module): Model to export
        output_path (Path): Path to save the TorchScript model
        
    Returns:
        bool: Whether the export was successful
    """
    try:
        # Trace the model
        scripted_model = torch.jit.script(model)
        
        # Save the traced model
        scripted_model.save(output_path)
        
        logger.info(f"Exported model to TorchScript: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting to TorchScript: {e}")
        return False

def export_model(model_type, export_format, model_path=None, output_dir=None):
    """
    Export a model to the specified format.
    
    Args:
        model_type (str): Type of model to export
        export_format (str): Format to export to
        model_path (str, optional): Path to model checkpoint
        output_dir (str, optional): Directory to save the exported model
        
    Returns:
        bool: Whether the export was successful
    """
    # Validate model type
    if model_type not in MODEL_TYPES:
        logger.error(f"Unsupported model type: {model_type}")
        return False
    
    # Validate export format
    if export_format not in EXPORT_FORMATS:
        logger.error(f"Unsupported export format: {export_format}")
        return False
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(f"./models/exported/{model_type}")
    else:
        output_dir = Path(output_dir) / model_type
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model = load_model(model_type, model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
    
    # Export model
    output_path = output_dir / f"{model_type}.{export_format}"
    
    if export_format == "onnx":
        success = export_to_onnx(model, model_type, output_path)
    elif export_format == "torchscript":
        success = export_to_torchscript(model, output_path)
    else:
        logger.error(f"Unsupported export format: {export_format}")
        return False
    
    # Create metadata file
    if success:
        metadata = {
            "model_type": model_type,
            "export_format": export_format,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "original_model_path": model_path
        }
        
        metadata_path = output_dir / f"{model_type}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created metadata file: {metadata_path}")
    
    return success

def main():
    """Parse arguments and start the export process."""
    parser = argparse.ArgumentParser(description="Export LexoRead models to different formats")
    
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_TYPES.keys()),
                        help="Type of model to export")
    
    parser.add_argument("--format", type=str, required=True,
                        choices=EXPORT_FORMATS,
                        help="Format to export to")
    
    parser.add_argument("--model_path", type=str,
                        help="Path to model checkpoint")
    
    parser.add_argument("--output", type=str,
                        help="Directory to save exported model")
    
    args = parser.parse_args()
    
    # Export model
    success = export_model(
        args.model,
        args.format,
        model_path=args.model_path,
        output_dir=args.output
    )
    
    if success:
        logger.info("Model export complete!")
    else:
        logger.error("Model export failed.")
        exit(1)

if __name__ == "__main__":
    main()
