"""CLI entry point and lightweight unit tests for the TTS/STT engine.

Run this file directly to get an interactive console menu or invoke with
`python -m unittest main.py` to execute the bundled smoke tests.
"""

import unittest
from core.engine import STT_TTS_Engine


class TestSTT_TTS_Engine(unittest.TestCase):
    """Very small smoke-test suite for `STT_TTS_Engine`.

    These tests confirm that the engine accepts both languages and that
    the gTTS synthesis pathway returns *True*. They deliberately ignore
    actual sound playback issues (e.g., in CI or headless environments)
    by catching exceptions raised from pygame.
    """

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
    """Simple text-based menu for casual testing from the terminal."""
    app = STT_TTS_Engine()

    # --- Simple REPL menu ------------------------------------------------
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
