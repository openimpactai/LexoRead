#!/usr/bin/env python3
"""
Model evaluation script for LexoRead.

This script evaluates the performance of LexoRead models on test datasets.
It supports evaluation of all model components and generates comprehensive reports.
"""

import os
import argparse
import logging
import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import models
from text_adaptation.model import DyslexiaTextAdapter
from ocr.model import DyslexiaOCR
from tts.model import DyslexiaTTS
from reading_level.model import ReadingLevelAssessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_evaluator")

def load_test_data(test_data_path):
    """
    Load test data from a file.
    
    Args:
        test_data_path (str): Path to the test data file
        
    Returns:
        dict: Test data
    """
    try:
        with open(test_data_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def evaluate_text_adaptation(model_path=None, test_data_path=None):
    """
    Evaluate text adaptation model.
    
    Args:
        model_path (str, optional): Path to the model weights
        test_data_path (str, optional): Path to the test data
        
    Returns:
        dict: Evaluation results
    """
    # Load model
    adapter = DyslexiaTextAdapter(model_path=model_path)
    
    # If no test data provided, try to load default test data
    if test_data_path is None:
        test_data_path = Path(__file__).parent / "text_adaptation" / "test_data.json"
    
    # Load test data
    test_data = load_test_data(test_data_path)
    if test_data is None:
        logger.error("Failed to load test data")
        return {
            "error": "Failed to load test data"
        }
    
    # Check if test data has the expected format
    if not isinstance(test_data, list) or not all(isinstance(item, dict) for item in test_data):
        logger.error("Invalid test data format")
        return {
            "error": "Invalid test data format"
        }
    
    # Evaluate
    results = {
        "adaptation_scores": [],
        "processing_times": [],
        "error_count": 0
    }
    
    for item in tqdm(test_data, desc="Evaluating Text Adaptation"):
        try:
            # Extract input text
            input_text = item.get("input_text", "")
            
            # Process text
            start_time = time.time()
            output = adapter.adapt(input_text)
            processing_time = time.time() - start_time
            
            # Store results
            results["processing_times"].append(processing_time)
            
            # Check if ground truth is available
            if "adaptation_score" in item:
                # Calculate adaptation score (0-100)
                ground_truth = item["adaptation_score"]
                predicted_score = adapter.get_readability_score(input_text)
                score_diff = abs(ground_truth - predicted_score)
                results["adaptation_scores"].append(score_diff)
        
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            results["error_count"] += 1
    
    # Calculate metrics
    if results["adaptation_scores"]:
        results["avg_score_diff"] = sum(results["adaptation_scores"]) / len(results["adaptation_scores"])
    else:
        results["avg_score_diff"] = None
    
    results["avg_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"]) if results["processing_times"] else None
    results["success_rate"] = 1.0 - (results["error_count"] / len(test_data))
    
    return results

def evaluate_ocr(model_path=None, test_data_path=None):
    """
    Evaluate OCR model.
    
    Args:
        model_path (str, optional): Path to the model weights
        test_data_path (str, optional): Path to the test data
        
    Returns:
        dict: Evaluation results
    """
    # Load model
    ocr = DyslexiaOCR(model_path=model_path)
    
    # If no test data provided, try to load default test data
    if test_data_path is None:
        test_data_path = Path(__file__).parent / "ocr" / "test_data.json"
    
    # Load test data
    test_data = load_test_data(test_data_path)
    if test_data is None:
        logger.error("Failed to load test data")
        return {
            "error": "Failed to load test data"
        }
    
    # Initialize metrics
    results = {
        "accuracy": [],
        "edit_distances": [],
        "processing_times": [],
        "error_count": 0
    }
    
    # Evaluate
    for item in tqdm(test_data, desc="Evaluating OCR"):
        try:
            # Get image path
            image_path = item.get("image_path", "")
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                results["error_count"] += 1
                continue
            
            # Extract text
            start_time = time.time()
            extracted_text = ocr.extract_text(image_path)
            processing_time = time.time() - start_time
            
            # Store processing time
            results["processing_times"].append(processing_time)
            
            # Check if ground truth is available
            if "ground_truth" in item:
                ground_truth = item["ground_truth"]
                
                # Calculate accuracy (exact match)
                accuracy = 1.0 if extracted_text == ground_truth else 0.0
                results["accuracy"].append(accuracy)
                
                # Calculate edit distance
                edit_distance = calculate_edit_distance(extracted_text, ground_truth)
                results["edit_distances"].append(edit_distance)
        
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            results["error_count"] += 1
    
    # Calculate metrics
    results["avg_accuracy"] = sum(results["accuracy"]) / len(results["accuracy"]) if results["accuracy"] else None
    results["avg_edit_distance"] = sum(results["edit_distances"]) / len(results["edit_distances"]) if results["edit_distances"] else None
    results["avg_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"]) if results["processing_times"] else None
    results["success_rate"] = 1.0 - (results["error_count"] / len(test_data))
    
    return results

def evaluate_tts(model_path=None, test_data_path=None):
    """
    Evaluate TTS model.
    
    Args:
        model_path (str, optional): Path to the model weights
        test_data_path (str, optional): Path to the test data
        
    Returns:
        dict: Evaluation results
    """
    # Load model
    tts = DyslexiaTTS(model_path=model_path)
    
    # If no test data provided, try to load default test data
    if test_data_path is None:
        test_data_path = Path(__file__).parent / "tts" / "test_data.json"
    
    # Load test data
    test_data = load_test_data(test_data_path)
    if test_data is None:
        logger.error("Failed to load test data")
        return {
            "error": "Failed to load test data"
        }
    
    # Initialize metrics
    results = {
        "processing_times": [],
        "audio_durations": [],
        "error_count": 0
    }
    
    # Create output directory for audio samples
    output_dir = Path(__file__).parent / "tts" / "evaluation_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    for i, item in enumerate(tqdm(test_data, desc="Evaluating TTS")):
        try:
            # Get text
            text = item.get("text", "")
            
            # Synthesize speech
            start_time = time.time()
            audio = tts.synthesize(text)
            processing_time = time.time() - start_time
            
            # Store metrics
            results["processing_times"].append(processing_time)
            results["audio_durations"].append(audio.get_duration())
            
            # Save audio sample
            audio.save(output_dir / f"sample_{i}.wav")
        
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            results["error_count"] += 1
    
    # Calculate metrics
    results["avg_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"]) if results["processing_times"] else None
    results["avg_audio_duration"] = sum(results["audio_durations"]) / len(results["audio_durations"]) if results["audio_durations"] else None
    results["success_rate"] = 1.0 - (results["error_count"] / len(test_data))
    
    return results

def evaluate_reading_level(model_path=None, test_data_path=None):
    """
    Evaluate reading level assessment model.
    
    Args:
        model_path (str, optional): Path to the model weights
        test_data_path (str, optional): Path to the test data
        
    Returns:
        dict: Evaluation results
    """
    # Load model
    assessor = ReadingLevelAssessor(model_path=model_path)
    
    # If no test data provided, try to load default test data
    if test_data_path is None:
        test_data_path = Path(__file__).parent / "reading_level" / "test_data.json"
    
    # Load test data
    test_data = load_test_data(test_data_path)
    if test_data is None:
        logger.error("Failed to load test data")
        return {
            "error": "Failed to load test data"
        }
    
    # Initialize metrics
    results = {
        "level_accuracy": [],
        "level_difference": [],
        "readability_difference": [],
        "processing_times": [],
        "error_count": 0
    }
    
    # Evaluate
    for item in tqdm(test_data, desc="Evaluating Reading Level"):
        try:
            # Get text
            text = item.get("text", "")
            
            # Assess reading level
            start_time = time.time()
            assessment = assessor.assess_text(text)
            processing_time = time.time() - start_time
            
            # Store processing time
            results["processing_times"].append(processing_time)
            
            # Check if ground truth is available
            if "ground_truth" in item:
                ground_truth = item["ground_truth"]
                
                # Calculate level accuracy
                if "level" in ground_truth:
                    level_accuracy = 1.0 if assessment.level == ground_truth["level"] else 0.0
                    results["level_accuracy"].append(level_accuracy)
                    
                    level_diff = abs(assessment.level - ground_truth["level"])
                    results["level_difference"].append(level_diff)
                
                # Calculate readability difference
                if "readability_score" in ground_truth:
                    readability_diff = abs(assessment.readability_score - ground_truth["readability_score"])
                    results["readability_difference"].append(readability_diff)
        
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            results["error_count"] += 1
    
    # Calculate metrics
    results["avg_level_accuracy"] = sum(results["level_accuracy"]) / len(results["level_accuracy"]) if results["level_accuracy"] else None
    results["avg_level_difference"] = sum(results["level_difference"]) / len(results["level_difference"]) if results["level_difference"] else None
    results["avg_readability_difference"] = sum(results["readability_difference"]) / len(results["readability_difference"]) if results["readability_difference"] else None
    results["avg_processing_time"] = sum(results["processing_times"]) / len(results["processing_times"]) if results["processing_times"] else None
    results["success_rate"] = 1.0 - (results["error_count"] / len(test_data))
    
    return results

def calculate_edit_distance(str1, str2):
    """
    Calculate the Levenshtein edit distance between two strings.
    
    Args:
        str1 (str): First string
        str2 (str): Second string
        
    Returns:
        int: Edit distance
    """
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]

def generate_report(results, output_dir=None):
    """
    Generate evaluation report.
    
    Args:
        results (dict): Evaluation results
        output_dir (str, optional): Directory to save the report
        
    Returns:
        str: Path to the generated report
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "evaluation_results"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"evaluation_report_{timestamp}.json"
    
    # Add timestamp to results
    results["timestamp"] = timestamp
    
    # Write report
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation report saved to {report_path}")
    
    # Generate charts
    generate_charts(results, output_dir, timestamp)
    
    return str(report_path)

def generate_charts(results, output_dir, timestamp):
    """
    Generate charts for evaluation results.
    
    Args:
        results (dict): Evaluation results
        output_dir (Path): Directory to save the charts
        timestamp (str): Timestamp for filenames
    """
    # Create charts directory
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Generate performance chart
    plt.figure(figsize=(10, 6))
    
    models = []
    success_rates = []
    
    for model, result in results.items():
        if isinstance(result, dict) and "success_rate" in result:
            models.append(model)
            success_rates.append(result["success_rate"] * 100)
    
    plt.bar(models, success_rates)
    plt.xlabel('Model')
    plt.ylabel('Success Rate (%)')
    plt.title('Model Success Rates')
    plt.ylim(0, 100)
    plt.savefig(charts_dir / f"success_rates_{timestamp}.png")
    
    # Generate processing time chart
    plt.figure(figsize=(10, 6))
    
    models = []
    processing_times = []
    
    for model, result in results.items():
        if isinstance(result, dict) and "avg_processing_time" in result and result["avg_processing_time"] is not None:
            models.append(model)
            processing_times.append(result["avg_processing_time"])
    
    plt.bar(models, processing_times)
    plt.xlabel('Model')
    plt.ylabel('Average Processing Time (s)')
    plt.title('Model Processing Times')
    plt.savefig(charts_dir / f"processing_times_{timestamp}.png")

def evaluate_all_models(models_dir=None, test_data_dir=None):
    """
    Evaluate all models.
    
    Args:
        models_dir (str, optional): Directory containing model weights
        test_data_dir (str, optional): Directory containing test data
        
    Returns:
        dict: Evaluation results for all models
    """
    results = {}
    
    # Determine model paths
    text_adaptation_model = os.path.join(models_dir, "text_adaptation", "text_adaptation_model.pt") if models_dir else None
    ocr_model = os.path.join(models_dir, "ocr", "ocr_model.pt") if models_dir else None
    tts_model = os.path.join(models_dir, "tts", "tts_model.pt") if models_dir else None
    reading_level_model = os.path.join(models_dir, "reading_level", "reading_level_model.pt") if models_dir else None
    
    # Determine test data paths
    text_adaptation_test = os.path.join(test_data_dir, "text_adaptation_test.json") if test_data_dir else None
    ocr_test = os.path.join(test_data_dir, "ocr_test.json") if test_data_dir else None
    tts_test = os.path.join(test_data_dir, "tts_test.json") if test_data_dir else None
    reading_level_test = os.path.join(test_data_dir, "reading_level_test.json") if test_data_dir else None
    
    # Evaluate text adaptation model
    try:
        logger.info("Evaluating Text Adaptation model")
        results["text_adaptation"] = evaluate_text_adaptation(text_adaptation_model, text_adaptation_test)
    except Exception as e:
        logger.error(f"Error evaluating Text Adaptation model: {e}")
        results["text_adaptation"] = {"error": str(e)}
    
    # Evaluate OCR model
    try:
        logger.info("Evaluating OCR model")
        results["ocr"] = evaluate_ocr(ocr_model, ocr_test)
    except Exception as e:
        logger.error(f"Error evaluating OCR model: {e}")
        results["ocr"] = {"error": str(e)}
    
    # Evaluate TTS model
    try:
        logger.info("Evaluating TTS model")
        results["tts"] = evaluate_tts(tts_model, tts_test)
    except Exception as e:
        logger.error(f"Error evaluating TTS model: {e}")
        results["tts"] = {"error": str(e)}
    
    # Evaluate Reading Level model
    try:
        logger.info("Evaluating Reading Level model")
        results["reading_level"] = evaluate_reading_level(reading_level_model, reading_level_test)
    except Exception as e:
        logger.error(f"Error evaluating Reading Level model: {e}")
        results["reading_level"] = {"error": str(e)}
    
    return results

def main():
    """Parse arguments and start the evaluation process."""
    parser = argparse.ArgumentParser(description="Evaluate LexoRead models")
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['text_adaptation', 'ocr', 'tts', 'reading_level', 'all'],
                        help='Model to evaluate')
    
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model weights')
    
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Evaluate models
    if args.model == 'all':
        results = evaluate_all_models(args.model_path, args.test_data)
    elif args.model == 'text_adaptation':
        results = {args.model: evaluate_text_adaptation(args.model_path, args.test_data)}
    elif args.model == 'ocr':
        results = {args.model: evaluate_ocr(args.model_path, args.test_data)}
    elif args.model == 'tts':
        results = {args.model: evaluate_tts(args.model_path, args.test_data)}
    elif args.model == 'reading_level':
        results = {args.model: evaluate_reading_level(args.model_path, args.test_data)}
    
    # Generate report
    generate_report(results, args.output_dir)

if __name__ == "__main__":
    main()
