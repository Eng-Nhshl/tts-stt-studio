import sys
import os
import tempfile
import uuid
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QTextEdit, QTabWidget, QMessageBox, QComboBox, QProgressBar, QFrame
)
import pygame
import speech_recognition as sr
from gtts import gTTS
from anomaly_detector import AnomalyDetector

# --- Engine stays the same except status_callback replaced with Qt signals ---

class STT_TTS_Engine:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.anomaly_detector = AnomalyDetector()
        print("TTS & STT engine initialized successfully")

    def text_to_speech(self, text, language="en", status_callback=None):
        try:
            if not text:
                if status_callback:
                    status_callback.emit("Error: No text provided for conversion")
                raise ValueError("No text provided for conversion")
            if self.anomaly_detector.detect_offensive_content(text, language):
                if status_callback:
                    status_callback.emit("Error: Text contains offensive content")
                raise ValueError("Text contains offensive content")
            print(f"Converting text to speech. Language: {language}, Text: {text}")
            tts = gTTS(text=text, lang=language)
            temp_filename = os.path.join(
                tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3"
            )
            try:
                tts.save(temp_filename)
                print(f"Playing speech from: {temp_filename}")
                pygame.mixer.init()
                try:
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    if status_callback:
                        status_callback.emit("Text to speech completed!")
                    print("Text-to-speech playback completed successfully")
                    return True
                except Exception:
                    if status_callback:
                        status_callback.emit("Error: Audio playback failed")
                    print("Audio file was created, but playback failed.")
                    return True
                finally:
                    pygame.mixer.quit()
            finally:
                for attempt in range(3):
                    try:
                        os.remove(temp_filename)
                        break
                    except Exception as e:
                        print(f"Attempt {attempt+1} to delete temp file failed: {str(e)}")
                        if attempt == 2:
                            print(f"Warning: Could not delete temp file {temp_filename} after 3 attempts")
                        else:
                            time.sleep(0.1)
            return True
        except ValueError as ve:
            print(f"Error in text_to_speech: {str(ve)}")
            return False
        except Exception as e:
            print(f"Error in text_to_speech: Unexpected error: {str(e)}")
            return False

    def speech_to_text(self, language="en", timeout=5, status_callback=None):
        try:
            if status_callback:
                status_callback.emit("Calibrating microphone...")
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                if status_callback:
                    status_callback.emit("Listening...")
                try:
                    audio = self.recognizer.listen(source, timeout=timeout)
                    if status_callback:
                        status_callback.emit("Processing speech...")
                    try:
                        audio_data = np.frombuffer(audio.get_raw_data(), np.int16)
                        audio_data = audio_data.astype(np.float32) / 32768.0
                        sample_rate = audio.sample_rate
                        if self.anomaly_detector.detect_audio_anomaly(audio_data, sample_rate):
                            raise ValueError("Audio contains suspicious patterns")
                    except Exception as e:
                        print(f"Error processing audio data: {str(e)}")
                        raise ValueError("Error processing audio data")
                    text = self.recognizer.recognize_google(audio, language=language)
                    if self.anomaly_detector.detect_offensive_content(text, language):
                        raise ValueError("Recognized text contains offensive content")
                    return text
                except sr.WaitTimeoutError:
                    if status_callback:
                        status_callback.emit("No speech detected within timeout period")
                    raise ValueError("No speech detected")
                except sr.UnknownValueError:
                    if status_callback:
                        status_callback.emit("Could not understand audio")
                    raise ValueError("Could not understand audio")
                except sr.RequestError as e:
                    if status_callback:
                        status_callback.emit(f"Speech recognition service error: {str(e)}")
                    raise ValueError(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            print(f"Error in speech_to_text: {str(e)}")
            return ""


# --- Worker threads for async operations ---

class TTSThread(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(bool)
    def __init__(self, engine, text, language):
        super().__init__()
        self.engine = engine
        self.text = text
        self.language = language

    def run(self):
        result = self.engine.text_to_speech(self.text, self.language, self.status)
        self.finished.emit(result)

class STTThread(QThread):
    status = pyqtSignal(str)
    result = pyqtSignal(str)
    def __init__(self, engine, language):
        super().__init__()
        self.engine = engine
        self.language = language

    def run(self):
        text = self.engine.speech_to_text(self.language, status_callback=self.status)
        self.result.emit(text)

# --- Main GUI ---

class GlassFrame(QFrame):
    """A semi-transparent frosted-glass effect QFrame."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            GlassFrame {
                background: rgba(255,255,255,0.15);
                border-radius: 24px;
                border: 1.5px solid rgba(255,255,255,0.22);
            }
        """)

class NeonButton(QPushButton):
    """A neon-style QPushButton."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QPushButton {
                color: #fff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d1cbe, stop:1 #00d2ff);
                border-radius: 18px;
                border: none;
                padding: 10px 28px;
                font-size: 1.2em;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:1, y1:0, x2:0, y2:1, stop:0 #00d2ff, stop:1 #3d1cbe);
            }
            QPushButton:pressed {
                background: #0c024d;
            }
        """)

class TitleLabel(QLabel):
    """A stylish glowing title label."""
    def __init__(self, text):
        super().__init__(text)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setStyleSheet("""
            QLabel {
                font-size: 2.2em;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                color: #fff;
                margin: 18px 0 0 0;
                font-weight: 600;
                letter-spacing: 1.5px;
                background: qlineargradient(x1:0 y1:0, x2:0 y2:1, stop:0 rgba(0,210,255,0.3), stop:1 rgba(61,28,190,0.3));
            }
        """)

class GlassTabWidget(QTabWidget):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
        QTabWidget::pane {
            border: none;
        }
        QTabBar::tab {
            background: rgba(255,255,255,0.08);
            color: #fff;
            border-top-left-radius: 16px;
            border-top-right-radius: 16px;
            min-height: 36px;
            min-width: 150px;
            font-weight: 600;
            font-size: 1.1em;
            margin-right: 4px;
            padding: 8px 24px;
        }
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d1cbe, stop:1 #00d2ff);
            color: #fff;
        }
        """)

class GlassComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
        QComboBox {
            background: rgba(255,255,255,0.18);
            color: #fff;
            border-radius: 12px;
            border: 1px solid #3d1cbe;
            padding: 6px 18px;
            font-size: 1.05em;
        }
        QComboBox QAbstractItemView {
            background: #24243e;
            color: #fff;
        }
        """)

class GlassTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
        QTextEdit {
            background: rgba(255,255,255,0.10);
            color: #fff;
            border-radius: 12px;
            border: 1.5px solid #00d2ff;
            font-size: 1.07em;
            padding: 8px 10px;
            min-height: 80px;
        }
        """)

class GlassLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setStyleSheet("""
        QLabel {
            color: #e0e5ff;
            font-size: 1.04em;
            margin-top: 12px;
            font-weight: 500;
            background: qlineargradient(x1:0 y1:0, x2:0 y2:1, stop:0 rgba(224,229,255,0.2), stop:1 rgba(224,229,255,0.1));
        }
        """)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.engine = STT_TTS_Engine()
        self.setWindowTitle("✨ AI Multilingual TTS & STT Studio ✨")
        self.setWindowIcon(QtGui.QIcon.fromTheme("microphone"))
        self.setGeometry(200, 120, 700, 530)
        self.setMinimumSize(600, 440)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(
                    x1:0 y1:0, x2:1 y2:1,
                    stop:0 #0f2027, stop:0.48 #2c5364, stop:1 #24243e
                );
            }
        """)
        self.init_ui()
        self.showNormal()

    def init_ui(self):
        # Title
        title = TitleLabel("✨ AI Multilingual TTS & STT Studio ✨")

        # Tabs
        self.tabs = GlassTabWidget()
        self.tab_tts = QWidget()
        self.tab_stt = QWidget()
        self.tabs.addTab(self.tab_tts, "Text to Speech")
        self.tabs.addTab(self.tab_stt, "Speech to Text")

        # --- TTS Tab ---
        tts_layout = QVBoxLayout()
        tts_card = GlassFrame()
        tts_card_layout = QVBoxLayout()
        self.tts_text = GlassTextEdit()
        self.tts_text.setPlaceholderText("Type your text here...")
        self.tts_language_combo = GlassComboBox()
        self.tts_language_combo.addItem("English", "en")
        self.tts_language_combo.addItem("Arabic", "ar")
        self.btn_tts = NeonButton("🔊 Convert to Speech")
        self.btn_tts.clicked.connect(self.on_tts)
        self.tts_status = GlassLabel("")
        tts_card_layout.addWidget(GlassLabel("Enter text:"))
        tts_card_layout.addWidget(self.tts_text)
        tts_card_layout.addWidget(GlassLabel("Language:"))
        tts_card_layout.addWidget(self.tts_language_combo)
        tts_card_layout.addWidget(self.btn_tts)
        tts_card_layout.addWidget(self.tts_status)
        tts_card_layout.addStretch()
        tts_card.setLayout(tts_card_layout)
        tts_layout.addWidget(tts_card)
        tts_layout.addStretch()
        self.tab_tts.setLayout(tts_layout)

        # --- STT Tab ---
        stt_layout = QVBoxLayout()
        stt_card = GlassFrame()
        stt_card_layout = QVBoxLayout()
        self.stt_language_combo = GlassComboBox()
        self.stt_language_combo.addItem("English", "en")
        self.stt_language_combo.addItem("Arabic", "ar")
        self.btn_stt = NeonButton("🎙️ Start Listening")
        self.btn_stt.clicked.connect(self.on_stt)
        self.stt_status = GlassLabel("")
        self.stt_result = GlassTextEdit()
        self.stt_result.setReadOnly(True)
        stt_card_layout.addWidget(GlassLabel("Language:"))
        stt_card_layout.addWidget(self.stt_language_combo)
        stt_card_layout.addWidget(self.btn_stt)
        stt_card_layout.addWidget(self.stt_status)
        stt_card_layout.addWidget(GlassLabel("Recognized Text:"))
        stt_card_layout.addWidget(self.stt_result)
        stt_card_layout.addStretch()
        stt_card.setLayout(stt_card_layout)
        stt_layout.addWidget(stt_card)
        stt_layout.addStretch()
        self.tab_stt.setLayout(stt_layout)

        # --- Main Layout ---
        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def on_tts(self):
        text = self.tts_text.toPlainText().strip()
        language = self.tts_language_combo.currentData()
        if not text:
            self.tts_status.setText("Please enter some text.")
            return
        self.btn_tts.setEnabled(False)
        self.tts_status.setText("Processing...")
        self.tts_thread = TTSThread(self.engine, text, language)
        self.tts_thread.status.connect(self.tts_status.setText)
        self.tts_thread.finished.connect(self.tts_finished)
        self.tts_thread.start()

    def tts_finished(self, result):
        self.btn_tts.setEnabled(True)
        if result:
            self.tts_status.setText("Text successfully converted to speech.")
        else:
            self.tts_status.setText("Text contains offensive content!")

    def on_stt(self):
        language = self.stt_language_combo.currentData()
        self.btn_stt.setEnabled(False)
        self.stt_status.setText("Listening...")
        self.stt_result.setPlainText("")
        self.stt_thread = STTThread(self.engine, language)
        self.stt_thread.status.connect(self.stt_status.setText)
        self.stt_thread.result.connect(self.stt_finished)
        self.stt_thread.start()

    def stt_finished(self, text):
        self.btn_stt.setEnabled(True)
        if text:
            self.stt_result.setPlainText(text)
            self.stt_status.setText("Speech recognized successfully.")
        else:
            self.stt_result.setPlainText("")
            self.stt_status.setText("No speech detected or Recognized text contains offensive content!")

# --- Main Entrypoint ---

def main():
    app = QApplication(sys.argv)
    # Set awesome font for all
    app.setFont(QtGui.QFont("Segoe UI", 11))
    # Optional: Add a window shadow for wow effect (on supported platforms)
    window = MainWindow()
    window.setWindowFlags(window.windowFlags() | QtCore.Qt.Window | QtCore.Qt.MSWindowsFixedSizeDialogHint)
    window.setAttribute(Qt.WA_TranslucentBackground, True)
    window.setAutoFillBackground(False)
    sys.exit(app.exec_())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())