import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QProgressBar,
)

# --- Import engine from core
from core.engine import STT_TTS_Engine

# Import custom widgets
from widgets import (
    GlassFrame,
    NeonButton,
    TitleLabel,
    GlassTabWidget,
    GlassComboBox,
    GlassTextEdit,
    GlassLabel,
)


class TTSThread(QThread):
    """Background thread that converts *text* to speech."""

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
    """Background thread that listens to the microphone and performs STT."""

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
class MainWindow(QWidget):
    """Top-level window that hosts two tabs: TTS and STT."""

    def __init__(self):
        super().__init__()
        self.engine = STT_TTS_Engine()
        self.setWindowTitle("✨ AI Multilingual TTS & STT Studio ✨")
        self.setWindowIcon(QtGui.QIcon.fromTheme("microphone"))
        self.setGeometry(200, 120, 700, 530)
        self.setMinimumSize(600, 440)
        self.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(
                    x1:0 y1:0, x2:1 y2:1,
                    stop:0 #0f2027, stop:0.48 #2c5364, stop:1 #24243e
                );
            }
        """
        )
        self.init_ui()
        self.showNormal()

    def init_ui(self):
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
        self.tts_progress = QProgressBar()
        self.tts_progress.setRange(0, 0)
        self.tts_progress.setTextVisible(False)
        self.tts_progress.hide()
        tts_card_layout.addWidget(GlassLabel("Enter text:"))
        tts_card_layout.addWidget(self.tts_text)
        tts_card_layout.addWidget(GlassLabel("Language:"))
        tts_card_layout.addWidget(self.tts_language_combo)
        tts_card_layout.addWidget(self.btn_tts)
        tts_card_layout.addWidget(self.tts_status)
        tts_card_layout.addWidget(self.tts_progress)
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
        self.stt_progress = QProgressBar()
        self.stt_progress.setRange(0, 0)
        self.stt_progress.setTextVisible(False)
        self.stt_progress.hide()
        self.stt_result = GlassTextEdit()
        self.stt_result.setReadOnly(True)
        stt_card_layout.addWidget(GlassLabel("Language:"))
        stt_card_layout.addWidget(self.stt_language_combo)
        stt_card_layout.addWidget(self.btn_stt)
        stt_card_layout.addWidget(self.stt_status)
        stt_card_layout.addWidget(self.stt_progress)
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
        self.tts_progress.show()
        self.tts_status.setText("Processing...")
        self.tts_thread = TTSThread(self.engine, text, language)
        self.tts_thread.status.connect(self.tts_status.setText)
        self.tts_thread.finished.connect(self.tts_finished)
        self.tts_thread.start()

    def tts_finished(self, result):
        self.btn_tts.setEnabled(True)
        self.tts_progress.hide()
        if result:
            self.tts_status.setText("Text successfully converted to speech.")
        else:
            current = self.tts_status.text()
            if not current.startswith("Error"):
                self.tts_status.setText("Could not convert text to speech.")

    def on_stt(self):
        language = self.stt_language_combo.currentData()
        self.btn_stt.setEnabled(False)
        self.stt_progress.show()
        self.stt_status.setText("Listening...")
        self.stt_result.setPlainText("")
        self.stt_thread = STTThread(self.engine, language)
        self.stt_thread.status.connect(self.stt_status.setText)
        self.stt_thread.result.connect(self.stt_finished)
        self.stt_thread.start()

    def stt_finished(self, text):
        self.btn_stt.setEnabled(True)
        self.stt_progress.hide()
        if text:
            self.stt_result.setPlainText(text)
            self.stt_status.setText("Speech recognized successfully.")
        else:
            self.stt_result.setPlainText("")
            current = self.stt_status.text()
            if not current.startswith("Error"):
                self.stt_status.setText("No speech detected.")


# --- Main Entrypoint ---
def main():
    app = QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 11))
    window = MainWindow()
    window.setWindowFlags(
        window.windowFlags() | QtCore.Qt.Window | QtCore.Qt.MSWindowsFixedSizeDialogHint
    )
    window.setAttribute(Qt.WA_TranslucentBackground, True)
    window.setAutoFillBackground(False)
    sys.exit(app.exec_())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
