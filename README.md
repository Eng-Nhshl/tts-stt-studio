# Multilingual TTS & STT Studio

A modern, feature-rich Text-to-Speech (TTS) and Speech-to-Text (STT) studio featuring:

* Multilingual support (English 🇬🇧 / Arabic 🇸🇦)
* Real-time anomaly & offensive-content detection powered by Isolation Forest + bad-word datasets
* Caching, persistent outputs, and both CLI & PyQt5 GUI front-ends
* Modular codebase – core engine, GUI, widgets, styles, config – for easy extension

---

## Quick start

```bash
# install deps
pip install -r requirements.txt

# run GUI
python stt_tts_gui.py

# or CLI
python main.py
```

Audio outputs land in `outputs/tts/` and transcriptions in `outputs/stt/`.  The `.cache/` folder stores hashed MP3s to avoid redundant requests to gTTS.

---

## Updated project structure

```
.
├── core/              # shared STT_TTS_Engine (business logic)
│   └── engine.py
├── stt_tts_gui.py     # PyQt5 front-end (thin – layout & signals only)
├── widgets.py         # reusable glass/neon widgets
├── styles.py          # centralized Qt style-sheet constants
├── anomaly_detector.py# audio/text anomaly + offensive-word checks
├── config.py          # logging + path constants (CACHE_DIR, OUTPUT_DIR …)
├── datasets/          # offensive words JSON (en.json / ar.json)
├── outputs/           # generated mp3 / transcripts
├── .cache/            # TTS cache (auto-created)
├── main.py            # CLI menu + smoke tests
└── requirements.txt
```

See comments/docstrings in each file for detailed explanations.


## Features

- ✨ Modern, responsive UI with a beautiful dark theme
- 🌐 Multilingual support (English and Arabic)
- 🎤 Real-time speech recognition
- 🎤 Text-to-speech conversion
- 🔍 Anomaly detection for both text and audio
- 🔄 Automatic microphone calibration
- 🎨 Customizable UI components with glass effects
- 📊 Real-time status updates

## Project Structure

```
TTS & STT/
├── __pycache__/            # Python bytecode cache
├── anomaly_detector.py     # Machine learning-based anomaly detection system
├── datasets/               # Directory for storing bad word datasets for languages models
├── main.py                # Main application entry point
├── requirements.txt        # Project dependencies
└── stt_tts_gui.py         # GUI implementation with custom components
```

## Technical Overview

### Core Components

1. **STT_TTS_Engine**
   - Handles core text-to-speech and speech-to-text functionality
   - Implements anomaly detection for both text and audio
   - Supports multiple languages
   - Handles temporary file management

2. **GUI Components**
   - Modern PyQt-based interface
   - Custom UI components with glass effects
   - Real-time status updates
   - Tabbed interface for different functionalities

3. **Advanced Anomaly Detection System**
   - Real-time audio anomaly detection using Isolation Forest algorithm
   - Text content analysis for offensive language
   - Multi-language support (English and Arabic)
   - Continuous learning from audio patterns
   - Statistical feature extraction for audio analysis
   - Customizable contamination thresholds
   - Automatic model initialization and updates

## Installation

1. Clone the repository:
```bash
git clone https://git@github.com:Eng-Nhshl/tts-stt-studio.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Launch the application for the console interface:
```bash
python main.py
```

2. Launch the application for the GUI interface:
```bash
python stt_tts_gui.py
```

3. Use the interface:
   - Select your preferred language (English/Arabic)
   - Use the "Speak" button for speech-to-text
   - Enter text and use "Convert" for text-to-speech
   - Monitor status updates in real-time

## Dependencies

- numpy==2.0.0
- numba==0.58.1
- scikit-learn>=0.24.0
- pyttsx3>=2.90
- SpeechRecognition>=3.8.1
- pyaudio>=0.2.11
- vosk>=0.3.42
- arabic-reshaper>=2.1.3
- python-bidi>=0.4.2
- scipy>=1.7.0
- librosa==0.10.1
- sounddevice>=0.4.4
- PyQt6>=6.4.0
- pyqtgraph>=0.13.1

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the SCC Student's License.

## Acknowledgments

- Thanks to all contributors and open-source libraries used in this project
- Special thanks to the speech recognition and text-to-speech communities
