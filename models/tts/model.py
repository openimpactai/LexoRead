#!/usr/bin/env python3
"""
Text-to-Speech Model for LexoRead

This module provides a model to convert text to speech with special focus
on clear, dyslexia-friendly pronunciation and customizable speech parameters.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import re
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tts_model")

class ConvBNBlock(nn.Module):
    """
    Convolutional block with batch normalization.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    """
    Text encoder for TTS model.
    """
    
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512):
        super(Encoder, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            ConvBNBlock(embedding_dim, hidden_dim, kernel_size=5, padding=2),
            ConvBNBlock(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            ConvBNBlock(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Transpose for conv layers [batch, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Convolutional layers
        conv_output = self.conv_layers(embedded)
        
        # Transpose back for LSTM [batch, seq_len, hidden_dim]
        conv_output = conv_output.transpose(1, 2)
        
        # LSTM
        lstm_output, _ = self.lstm(conv_output)
        
        return lstm_output

class LocationLayer(nn.Module):
    """
    Location-sensitive attention layer.
    """
    
    def __init__(self, attention_dim, attention_filters, attention_kernel_size):
        super(LocationLayer, self).__init__()
        
        self.location_conv = nn.Conv1d(
            in_channels=2,  # Previous attention weights + cumulative attention weights
            out_channels=attention_filters,
            kernel_size=attention_kernel_size,
            padding=(attention_kernel_size - 1) // 2,
            bias=False
        )
        self.location_dense = nn.Linear(attention_filters, attention_dim, bias=False)
        
    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    """
    Attention mechanism for TTS.
    """
    
    def __init__(self, attention_dim, encoder_dim, decoder_dim, attention_filters=32, attention_kernel_size=31):
        super(Attention, self).__init__()
        
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.location_layer = LocationLayer(attention_dim, attention_filters, attention_kernel_size)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, query, memory, attention_weights_cat):
        """
        Calculate attention weights.
        
        Args:
            query: Decoder state
            memory: Encoder outputs
            attention_weights_cat: Previous and cumulative attention weights
            
        Returns:
            attention_weights: Current attention weights
            context: Context vector
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_memory = self.memory_layer(memory)
        processed_attention = self.location_layer(attention_weights_cat)
        
        # Calculate attention scores
        energies = self.v(torch.tanh(processed_query + processed_memory + processed_attention))
        energies = energies.squeeze(-1)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(energies, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), memory)
        context = context.squeeze(1)
        
        return attention_weights, context

class Decoder(nn.Module):
    """
    Decoder for TTS model.
    """
    
    def __init__(self, encoder_dim, attention_dim, decoder_dim, mel_dim, r):
        super(Decoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.mel_dim = mel_dim
        self.r = r  # Reduction factor
        
        # Attention
        self.attention = Attention(attention_dim, encoder_dim, decoder_dim)
        
        # Decoder LSTM
        self.prenet = nn.Sequential(
            nn.Linear(mel_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim),
            nn.LSTMCell(decoder_dim, decoder_dim)
        ])
        
        # Output layers
        self.linear_projection = nn.Linear(decoder_dim + encoder_dim, mel_dim * r)
        
        # Initialize attention states
        self.attention_weights = None
        self.attention_weights_cum = None
        self.context = None
        
    def initialize_states(self, encoder_outputs, batch_size):
        """
        Initialize decoder states.
        """
        # Initialize attention states
        self.attention_weights = torch.zeros(batch_size, encoder_outputs.size(1), device=encoder_outputs.device)
        self.attention_weights_cum = torch.zeros(batch_size, encoder_outputs.size(1), device=encoder_outputs.device)
        self.context = torch.zeros(batch_size, self.encoder_dim, device=encoder_outputs.device)
        
        # Initialize LSTM states
        h_list = [torch.zeros(batch_size, self.decoder_dim, device=encoder_outputs.device) for _ in range(2)]
        c_list = [torch.zeros(batch_size, self.decoder_dim, device=encoder_outputs.device) for _ in range(2)]
        
        return h_list, c_list
        
    def forward(self, encoder_outputs, mel_targets=None, teacher_forcing_ratio=1.0):
        """
        Forward pass of the decoder.
        
        Args:
            encoder_outputs: Outputs from the encoder
            mel_targets: Target mel spectrograms (for teacher forcing)
            teacher_forcing_ratio: Teacher forcing ratio (0-1)
            
        Returns:
            mel_outputs: Predicted mel spectrograms
            attention_weights: All attention weights
        """
        batch_size = encoder_outputs.size(0)
        max_len = mel_targets.size(1) // self.r if mel_targets is not None else 200
        
        # Initialize states
        h_list, c_list = self.initialize_states(encoder_outputs, batch_size)
        
        # Initialize input (zero frame)
        decoder_input = torch.zeros(batch_size, self.mel_dim, device=encoder_outputs.device)
        
        mel_outputs = []
        attention_weights_list = []
        
        for t in range(max_len):
            # Prepare attention inputs
            attention_weights_cat = torch.cat([
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1)
            ], dim=1)
            
            # Apply attention
            self.attention_weights, self.context = self.attention(
                h_list[-1], encoder_outputs, attention_weights_cat
            )
            
            # Update cumulative attention weights
            self.attention_weights_cum = self.attention_weights_cum + self.attention_weights
            
            # Apply prenet to input
            prenet_output = self.prenet(decoder_input)
            
            # LSTM cells
            lstm_input = torch.cat([prenet_output, self.context], dim=1)
            h_list[0], c_list[0] = self.lstm_cells[0](lstm_input, (h_list[0], c_list[0]))
            h_list[1], c_list[1] = self.lstm_cells[1](h_list[0], (h_list[1], c_list[1]))
            
            # Project to mel output
            projection_input = torch.cat([h_list[-1], self.context], dim=1)
            output = self.linear_projection(projection_input)
            
            # Reshape output (r frames)
            output = output.view(batch_size, self.mel_dim, self.r)
            
            mel_outputs.append(output)
            attention_weights_list.append(self.attention_weights)
            
            # Update input for next timestep
            if mel_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing
                next_input_idx = (t + 1) * self.r
                if next_input_idx < mel_targets.size(1):
                    decoder_input = mel_targets[:, next_input_idx - 1, :]
                else:
                    decoder_input = output[:, :, -1]
            else:
                # Use own prediction
                decoder_input = output[:, :, -1]
        
        # Concatenate outputs
        mel_outputs = torch.cat(mel_outputs, dim=2)
        mel_outputs = mel_outputs.transpose(1, 2)  # [batch, time, mel_dim]
        
        # Concatenate attention weights
        attention_weights_list = torch.stack(attention_weights_list, dim=1)  # [batch, time, enc_time]
        
        return mel_outputs, attention_weights_list

class Postnet(nn.Module):
    """
    Postnet to refine mel spectrogram predictions.
    """
    
    def __init__(self, mel_dim, hidden_dim=512, kernel_size=5, num_layers=5):
        super(Postnet, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        ))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh()
            ))
        
        # Last layer
        self.layers.append(nn.Sequential(
            nn.Conv1d(hidden_dim, mel_dim, kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm1d(mel_dim)
        ))
        
    def forward(self, x):
        """
        Forward pass of the postnet.
        
        Args:
            x: Mel spectrogram [batch, time, mel_dim]
            
        Returns:
            Refined mel spectrogram
        """
        x = x.transpose(1, 2)  # [batch, mel_dim, time]
        
        for layer in self.layers:
            x = layer(x)
            
        x = x.transpose(1, 2)  # [batch, time, mel_dim]
        return x

class DyslexiaTTSModel(nn.Module):
    """
    Full TTS model combining encoder, decoder, and postnet.
    """
    
    def __init__(self, vocab_size, mel_dim=80, encoder_dim=512, decoder_dim=512, attention_dim=128, r=1):
        super(DyslexiaTTSModel, self).__init__()
        
        self.encoder = Encoder(vocab_size, embedding_dim=512, hidden_dim=encoder_dim)
        self.decoder = Decoder(encoder_dim, attention_dim, decoder_dim, mel_dim, r)
        self.postnet = Postnet(mel_dim)
        
        self.mel_dim = mel_dim
        self.r = r
        
    def forward(self, text, mel_targets=None, teacher_forcing_ratio=1.0):
        """
        Forward pass of the model.
        
        Args:
            text: Input text tokens [batch, seq_len]
            mel_targets: Target mel spectrograms [batch, seq_len, mel_dim]
            teacher_forcing_ratio: Teacher forcing ratio (0-1)
            
        Returns:
            mel_outputs: Initial mel spectrogram predictions
            mel_outputs_postnet: Refined mel spectrogram predictions
            attention_weights: Attention weights
        """
        # Encode text
        encoder_outputs = self.encoder(text)
        
        # Decode to mel spectrogram
        mel_outputs, attention_weights = self.decoder(
            encoder_outputs, mel_targets, teacher_forcing_ratio
        )
        
        # Refine with postnet
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        
        return mel_outputs, mel_outputs_postnet, attention_weights

class AudioProcessor:
    """
    Audio preprocessing and postprocessing utilities.
    """
    
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, 
                 mel_dim=80, mel_fmin=0.0, mel_fmax=8000.0):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            mel_dim: Number of mel bins
            mel_fmin: Minimum frequency for mel filter bank
            mel_fmax: Maximum frequency for mel filter bank
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_dim = mel_dim
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        
    def mel_spectrogram(self, y):
        """
        Generate mel spectrogram from audio signal.
        
        Args:
            y: Audio signal [time]
            
        Returns:
            mel_spectrogram: Mel spectrogram [time, mel_dim]
        """
        # STFT
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        
        # Power spectrogram
        S = np.abs(D) ** 2
        
        # Mel spectrogram
        mel = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.mel_dim,
                                  fmin=self.mel_fmin, fmax=self.mel_fmax)
        S_mel = np.dot(mel, S)
        
        # Log mel spectrogram
        log_S_mel = np.log10(np.maximum(1e-5, S_mel))
        
        return log_S_mel.T  # [time, mel_dim]
    
    def griffin_lim(self, mel_spectrogram, n_iter=50):
        """
        Griffin-Lim algorithm to convert mel spectrogram back to audio.
        """
        # Convert mel to linear spectrogram
        mel = librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.mel_dim,
                                  fmin=self.mel_fmin, fmax=self.mel_fmax)
        mel_basis = np.linalg.pinv(mel)
        linear_spec = np.dot(mel_basis, 10 ** mel_spectrogram.T)
        
        # Griffin-Lim
        y = librosa.griffinlim(
            linear_spec, n_iter=n_iter, hop_length=self.hop_length, win_length=self.win_length
        )
        
        return y

class DyslexiaTTS:
    """
    Main class for text-to-speech synthesis.
    
    This class provides methods to load the model, process text,
    and synthesize speech.
    """
    
    def __init__(self, model_path=None, device=None, vocab_path=None):
        """
        Initialize the TTS processor.
        
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
        self.model = DyslexiaTTSModel(vocab_size=len(self.vocab))
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No model path provided or model not found. Using untrained model.")
            
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
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
            for i, char in enumerate(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-'\"()[]{}"):
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
                "pronunciation_emphasis": 1.0,
                "speech_rate": 1.0,
                "pitch_variation": 1.0,
                "sample_rate": 22050,
                "vocoder": "griffin_lim",
                "voices": {
                    "default": {
                        "gender": "female",
                        "pitch_offset": 0.0,
                        "speech_rate_offset": 0.0
                    },
                    "male": {
                        "gender": "male",
                        "pitch_offset": -0.3,
                        "speech_rate_offset": -0.05
                    },
                    "child": {
                        "gender": "child",
                        "pitch_offset": 0.3,
                        "speech_rate_offset": 0.1
                    }
                },
                "dyslexia_adaptations": {
                    "word_pause_multiplier": 1.2,
                    "sentence_pause_multiplier": 1.5,
                    "complex_word_emphasis": 1.3,
                    "complex_word_length": 8
                }