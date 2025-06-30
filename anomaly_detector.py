import re
import os
import json
from typing import Dict, Set
from sklearn.ensemble import IsolationForest
import numpy as np
import librosa
from collections import deque, defaultdict
import time
import warnings
import random

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, n_samples=100, contamination=0.1):
        """Initialize the anomaly detector.
        
        Args:
            n_samples: Number of samples to maintain in history
            contamination: Expected proportion of outliers in the data
        """
        self.n_samples = n_samples
        self.contamination = contamination
        self.audio_features = deque(maxlen=n_samples)
        self.text_features = deque(maxlen=n_samples)
        
        # Initialize models with default parameters
        self.audio_model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        self.text_model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Initialize with some default samples
        self._initialize_models()
        self.offensive_words = self.load_offensive_words()
        
        # Ensure models are properly initialized
        if not hasattr(self.audio_model, 'is_fitted_'):
            self._initialize_models()
            print("Models re-initialized successfully")
        
    def extract_audio_features(self, audio_data, sample_rate):
        """Extract features from audio data.
        
        Args:
            audio_data: Array of audio samples
            sample_rate: Sampling rate of the audio
            
        Returns:
            np.array: Feature vector
        """
        try:
            # Extract basic audio features
            features = []
            
            # Add basic statistics
            features.extend([
                np.mean(audio_data),
                np.std(audio_data),
                np.max(audio_data),
                np.min(audio_data)
            ])
            
            return np.array(features)
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return None
    
    def extract_text_features(self, text, language='en'):
        """Extract features from text data.
        
        Args:
            text: Input text
            language: Language of the text
            
        Returns:
            np.array: Extracted features
        """
        try:
            # Clean text by removing special characters
            if language == 'en':
                # For English, keep only letters, numbers, spaces, and basic punctuation
                cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.!?')
            else:  # Arabic
                # For Arabic, keep only Arabic letters, spaces, and basic punctuation
                arabic_chars = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويآإأؤئةىةًٌٍَُِّْ'
                cleaned_text = ''.join(c for c in text if c in arabic_chars or c.isspace() or c in '.!?')
            
            # If text is empty after cleaning, return None
            if not cleaned_text.strip():
                return None
                
            # Basic text features
            features = [
                len(cleaned_text),  # Text length
                len(cleaned_text.split()),  # Word count
                cleaned_text.count("."),  # Sentence count
                cleaned_text.count("!"),  # Exclamation count
                cleaned_text.count("?"),  # Question count
                sum(1 for c in cleaned_text if c.isupper()) / len(cleaned_text) if len(cleaned_text) > 0 else 0,  # Uppercase ratio
                sum(1 for c in cleaned_text if c.isnumeric()) / len(cleaned_text) if len(cleaned_text) > 0 else 0,  # Numeric ratio
                0  # Offensive content flag (we'll handle this in detect_text_anomaly)
            ]
            
            return np.array(features)
        except Exception as e:
            print(f"Error extracting text features: {str(e)}")
            return None
    
    def detect_audio_anomaly(self, audio_data, sample_rate):
        """Detect anomalies in audio data.
        
        Args:
            audio_data: Array of audio samples
            sample_rate: Sampling rate of the audio
            
        Returns:
            bool: True if anomaly detected, False otherwise
        """
        try:
            # Calculate basic statistics
            mean = np.mean(np.abs(audio_data))
            std = np.std(audio_data)
            max_val = np.max(np.abs(audio_data))
            
            # Define thresholds for normal audio
            # These thresholds are more lenient to handle background noise
            if (mean < 0.05 and  # Low mean amplitude
                std < 0.1 and     # Low standard deviation
                max_val < 0.5):   # Low peak amplitude
                return False  # Likely just background noise
                
            # Extract features
            features = self.extract_audio_features(audio_data, sample_rate)
            if features is None:
                return False
                
            self.audio_features.append(features)
            
            # Retrain model if we have enough samples
            if len(self.audio_features) >= self.n_samples:
                try:
                    # Convert deque to numpy array
                    features_array = np.array(self.audio_features)
                    # Fit the model with updated samples
                    self.audio_model.fit(features_array)
                except Exception as e:
                    print(f"Error retraining audio model: {str(e)}")
            
            # Predict anomaly
            prediction = self.audio_model.predict([features])
            return prediction[0] == -1  # -1 indicates anomaly
        except Exception as e:
            print(f"Error in audio anomaly detection: {str(e)}")
            return False

    def _initialize_models(self):
        """Initialize Isolation Forest models with initial samples."""
        try:
            # Generate initial samples for text model
            text_samples = []
            for _ in range(self.n_samples):
                # Create sample text features
                text_samples.append([
                    random.randint(1, 100),  # Length
                    random.randint(1, 20),   # Word count
                    random.randint(0, 5),    # Punctuation count
                    random.randint(0, 5),    # Question count
                    random.random(),         # Uppercase ratio
                    random.random(),         # Numeric ratio
                    1 if random.random() < 0.1 else 0,  # Offensive content flag
                    random.random()          # Additional feature
                ])
            
            # Convert to numpy array
            text_samples = np.array(text_samples)
            
            # Initialize text model with more lenient contamination
            self.text_model = IsolationForest(
                n_estimators=100,
                contamination=0.1,  # Increased contamination to be more lenient
                random_state=42
            )
            self.text_model.fit(text_samples)
            
            # Generate initial samples for audio model
            audio_samples = []
            for _ in range(self.n_samples):
                # Create sample audio features with more realistic values for normal speech
                audio_samples.append([
                    0.01 + random.random() * 0.1,  # Mean (0.01-0.11)
                    0.05 + random.random() * 0.15,  # Std (0.05-0.20)
                    0.1 + random.random() * 0.4,   # Max (0.1-0.5)
                    -0.1 + random.random() * 0.4   # Min (-0.1-0.3)
                ])
            
            # Convert to numpy array
            audio_samples = np.array(audio_samples)
            
            # Initialize audio model with more lenient contamination
            self.audio_model = IsolationForest(
                n_estimators=100,
                contamination=0.1,  # Increased contamination to be more lenient
                random_state=42
            )
            self.audio_model.fit(audio_samples)
            
            print("Isolation Forest models initialized successfully")
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
        
    def load_offensive_words(self) -> Dict[str, Set[str]]:
        """Load offensive words from JSON files.
        
        Returns:
            Dict mapping language to set of offensive words
        """
        offensive_words = defaultdict(set)
        datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
        
        # Load English words
        try:
            with open(os.path.join(datasets_dir, 'en.json'), 'r', encoding='utf-8') as f:
                english_words = json.load(f)
                for word in english_words:
                    offensive_words['en'].add(word.lower())
        except Exception as e:
            print(f"Warning: Could not load English offensive words: {str(e)}")

        # Load Arabic words
        try:
            with open(os.path.join(datasets_dir, 'ar.json'), 'r', encoding='utf-8') as f:
                arabic_words = json.load(f)
                for word in arabic_words:
                    offensive_words['ar'].add(word.lower())
        except Exception as e:
            print(f"Warning: Could not load Arabic offensive words: {str(e)}")

        # Add common variations for English words
        if 'en' in offensive_words:
            variations = set()
            for word in offensive_words['en']:
                # Add common leetspeak variations
                variations.add(word.replace('e', '3'))
                variations.add(word.replace('a', '@'))
                variations.add(word.replace('o', '0'))
                variations.add(word.replace('i', '1'))
                variations.add(word.replace('s', '$'))
                
                # Add common misspellings
                variations.add(word.replace('ck', 'x'))
                variations.add(word.replace('ph', 'f'))
                variations.add(word.replace('th', 'z'))
            
            offensive_words['en'].update(variations)
        
        return offensive_words

    def detect_offensive_content(self, text: str, language: str = 'en') -> bool:
        """Detect offensive content in text.
        
        Args:
            text: Input text to check
            language: Language of the text ('en' or 'ar')
            
        Returns:
            bool: True if offensive content detected, False otherwise
        """
        if language not in self.offensive_words:
            return False
            
        # Normalize text
        text = text.lower()
        
        # Check for offensive words
        for word in self.offensive_words[language]:
            if word in text:
                return True
        
        return False

    def detect_text_anomaly(self, text, language='en'):
        """Detect anomalies in text data.
        
        Args:
            text: Input text
            language: Language of the text ('en' or 'ar')
            
        Returns:
            bool: True if anomaly detected, False otherwise
        """
        # For Arabic text, be more lenient with character checks
        if language == 'ar':
            # Arabic text can contain special characters like diacritics
            arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهويآإأؤءةًٌٍَُِّْ')
            if not any(c in arabic_chars for c in text):
                return True  # No Arabic characters found
            return False  # Arabic text is considered normal
            
        # For English text, allow common special characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!? \'"-')
        special_chars = set(text) - allowed_chars
        
        # Only flag as suspicious if there are unusual special characters
        if special_chars:
            # Check if the special characters are common punctuation or whitespace
            if not any(c in "\n\t\r\x0b\x0c" for c in special_chars):
                return False  # Common special characters are allowed
            return True  # Unusual special characters found
            
        features = self.extract_text_features(text, language)
        if features is None:
            return False
            
        self.text_features.append(features)
        
        # Retrain model if we have enough samples
        if len(self.text_features) >= self.n_samples:
            try:
                # Convert deque to numpy array
                features_array = np.array(self.text_features)
                # Fit the model with updated samples
                self.text_model.fit(features_array)
            except Exception as e:
                print(f"Error retraining text model: {str(e)}")
        
        # Predict anomaly
        prediction = self.text_model.predict([features])
        return prediction[0] == -1  # -1 indicates anomaly
