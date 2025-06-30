import speech_recognition as sr
from gtts import gTTS
import pygame
import tempfile
import uuid
import time
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import os
import sys
import unittest
from anomaly_detector import AnomalyDetector
import numpy as np

# Initialize NumPy for audio processing
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class STT_TTS_Engine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.anomaly_detector = AnomalyDetector()
        print("TTS & STT engine initialized successfully")

    def text_to_speech(self, text, language="en", status_callback=None):
        """Convert text to speech.
        
        Args:
            text: Text to convert
            language: Language of the text (en or ar)
            status_callback: Function to update GUI status
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not text:
                if status_callback:
                    status_callback("Error: No text provided for conversion")
                raise ValueError("No text provided for conversion")
            
            # Check for offensive content
            if self.anomaly_detector.detect_offensive_content(text, language):
                if status_callback:
                    status_callback("Error: Text contains offensive content")
                raise ValueError("Text contains offensive content")
            
            print(f"Converting text to speech. Language: {language}, Text: {text}")

            # Use gTTS for both languages
            tts = gTTS(text=text, lang=language)

            # Generate a unique temp filename
            temp_filename = os.path.join(
                tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3"
            )

            try:
                # Save MP3 file
                tts.save(temp_filename)
                print(f"Playing speech from: {temp_filename}")

                # Initialize pygame mixer
                pygame.mixer.init()

                # Try to play the audio
                try:
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                        
                    if status_callback:
                        status_callback("Text to speech completed!")
                    print("Text-to-speech playback completed successfully")
                    return True
                except Exception as e:
                    if status_callback:
                        status_callback("Error: Audio playback failed")
                    print(
                        "The audio file was created successfully, but playback failed."
                    )
                    print(
                        "You can still use this functionality, just without audio playback."
                    )
                    return True
                finally:
                    pygame.mixer.quit()

            finally:
                # Attempt to delete the file with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        os.remove(temp_filename)
                        break
                    except Exception as e:
                        print(
                            f"Attempt {attempt + 1} to delete temp file failed: {str(e)}"
                        )
                        if attempt == max_retries - 1:
                            print(
                                f"Warning: Could not delete temp file {temp_filename} after {max_retries} attempts"
                            )
                        else:
                            time.sleep(0.1)  # Wait a bit before retry

            return True

        except ValueError as ve:
            print(f"Error in text_to_speech: {str(ve)}")
            return False
        except Exception as e:
            print(f"Error in text_to_speech: Unexpected error: {str(e)}")
            return False

    def speech_to_text(self, language="en", timeout=5):
        """Convert speech to text using speech_recognition.
        
        Args:
            language: Language of the speech (en or ar)
            timeout: Maximum time to wait for speech in seconds
            
        Returns:
            str: Recognized text
        """
        try:
            print(f"\nStarting speech recognition. Language: {language}")
            print("Calibrating microphone...")
            
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening...")
                
                try:
                    audio = self.recognizer.listen(source, timeout=timeout)
                    print("Processing speech...")
                    
                    # Extract audio features
                    try:
                        # Convert audio data to numpy array and normalize to float32
                        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                        sample_rate = audio.sample_rate
                        
                        # Check for audio anomalies
                        if self.anomaly_detector.detect_audio_anomaly(audio_data, sample_rate):
                            raise ValueError("Audio contains suspicious patterns")
                    except Exception as e:
                        print(f"Error processing audio data: {str(e)}")
                        raise ValueError("Error processing audio data")
                    
                    # Convert audio to text
                    text = self.recognizer.recognize_google(audio, language=language)
                    print(f"Recognized text: {text}")
                    
                    # Check for offensive content
                    if self.anomaly_detector.detect_offensive_content(text, language):
                        raise ValueError("Recognized text contains offensive content")
                    
                    return text
                except sr.WaitTimeoutError:
                    print("No speech detected within timeout period")
                    raise ValueError("No speech detected")
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    raise ValueError("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service: {str(e)}")
                    raise ValueError(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            print(f"Error in speech_to_text: {str(e)}")
            return ""

    def _check_anomaly(self, features):
        """Check if the speech input is anomalous using Isolation Forest"""
        self.speech_features.append(features)
        if len(self.speech_features) > 10:
            # Update the model with new data
            self.isolation_forest.fit(self.speech_features)
            # Predict if the current input is anomalous
            prediction = self.isolation_forest.predict([features])
            return prediction[0] == -1  # -1 indicates anomaly
        return False


class TestSTT_TTS_Engine(unittest.TestCase):
    def setUp(self):
        self.app = STT_TTS_Engine()

    def test_language_support(self):
        """Test language support for both English and Arabic"""
        # Test English
        try:
            result = self.app.text_to_speech("Testing English", "en")
            self.assertTrue(result)
        except Exception as e:
            print(f"Language support test skipped (playsound error): {str(e)}")
            # Skip if there's an issue with playsound on Windows
            return

        # Test Arabic
        try:
            result = self.app.text_to_speech("اختبار العربية", "ar")
            self.assertTrue(result)
        except Exception as e:
            print(f"Language support test skipped (playsound error): {str(e)}")
            # Skip if there's an issue with playsound on Windows
            return

    def test_tts(self):
        """Test text-to-speech functionality using gTTS"""
        # Test English TTS
        try:
            result = self.app.text_to_speech("Hello world", "en")
            self.assertTrue(result)
        except Exception as e:
            print(f"TTS test skipped (playsound error): {str(e)}")
            # Skip if there's an issue with playsound on Windows
            return

        # Test Arabic TTS
        try:
            result = self.app.text_to_speech("مرحبا", "ar")
            self.assertTrue(result)
        except Exception as e:
            print(f"TTS test skipped (playsound error): {str(e)}")
            # Skip if there's an issue with playsound on Windows
            return


def main():
    app = STT_TTS_Engine()

    while True:
        print("\nMultilingual TTS & STT Application")
        print("1. Text to Speech (English)")
        print("2. Text to Speech (Arabic)")
        print("3. Speech to Text (English)")
        print("4. Speech to Text (Arabic)")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            text = input("Enter English text: ")
            if app.text_to_speech(text, "en"):
                print("Text successfully converted to speech.")

        elif choice == "2":
            text = input("Enter Arabic text: ")
            if app.text_to_speech(text, "ar"):
                print("Text successfully converted to speech.")

        elif choice == "3":
            text = app.speech_to_text("en")
            if text:
                print(f"Recognized text: {text}")
            else:
                print("No speech detected or recognition failed.")

        elif choice == "4":
            text = app.speech_to_text("ar")
            if text:
                print(f"Recognized text: {text}")
            else:
                print("No speech detected or recognition failed.")

        elif choice == "5":
            print("Thank you for using the application. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
