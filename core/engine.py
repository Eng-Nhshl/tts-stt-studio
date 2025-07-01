from __future__ import annotations

import os
import tempfile
import time
import uuid
import logging
from typing import Callable, Optional

import numpy as np
import pygame
import speech_recognition as sr
from gtts import gTTS
from arabic_reshaper import reshape
from bidi.algorithm import get_display

from anomaly_detector import AnomalyDetector
from config import logging  # sets up root logger


np.set_printoptions(precision=3, suppress=True)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _notify(cb: Optional[Callable[[str], None]], message: str) -> None:
    """Send *message* through *cb* if given.

    *cb* may be a plain Python callable or a Qt `pyqtSignal`; we detect the
    latter by the presence of an ``emit`` attribute.
    """
    if cb is None:
        return
    if hasattr(cb, "emit"):
        cb.emit(message)  # type: ignore[attr-defined]
    else:
        cb(message)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class STT_TTS_Engine:
    """Multilingual Speech-to-Text / Text-to-Speech engine."""

    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.anomaly_detector = AnomalyDetector()
        logger.info("TTS & STT engine initialized successfully")

    # ---------------------------------------------------------------------
    # Text -> Speech
    # ---------------------------------------------------------------------

    def text_to_speech(
        self,
        text: str,
        language: str = "en",
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Convert *text* into audible speech."""
        try:
            if not text:
                _notify(status_callback, "Error: No text provided for conversion!")
                raise ValueError("No text provided for conversion!")

            # Check for offensive content
            if self.anomaly_detector.detect_offensive_content(text, language):
                _notify(status_callback, "Error: Text contains offensive content!")
                raise ValueError("Text contains offensive content!")

            # Remove special characters that should be ignored when speaking
            import re

            cleaned_text = re.sub(r"[!@#$%^&*()_+]+", " ", text)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            # Remove text not matching the selected language script
            if language == "ar":
                # Strip Latin letters when speaking Arabic
                cleaned_text = re.sub(r"[A-Za-z]+", " ", cleaned_text)
            elif language == "en":
                # Strip Arabic letters when speaking English
                cleaned_text = re.sub(r"[\u0600-\u06FF]+", " ", cleaned_text)

            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            if not cleaned_text:
                _notify(
                    status_callback,
                    "Error: No speakable content after language filtering!",
                )
                raise ValueError("No speakable content after language filtering!")

            logger.info(
                "Converting text to speech. Language: %s, Original: %s, Cleaned: %s",
                language,
                text,
                cleaned_text,
            )

            from hashlib import sha1
            from config import CACHE_DIR

            # Deterministic cache key by language+cleaned_text
            cache_key = sha1(f"{language}:{cleaned_text}".encode("utf-8")).hexdigest()
            cached_mp3 = CACHE_DIR / f"{cache_key}.mp3"

            if cached_mp3.exists():
                logger.debug("Cache hit for TTS request (lang=%s)", language)
                temp_filename = str(cached_mp3)
            else:
                logger.debug("Cache miss for TTS request (lang=%s)", language)
                # Generate in temp dir then move to cache
                temp_filename = os.path.join(
                    tempfile.gettempdir(), f"tts_{uuid.uuid4()}.mp3"
                )
                # Synthesize speech and save
                gTTS(text=cleaned_text, lang=language).save(temp_filename)
                try:
                    import shutil

                    shutil.move(temp_filename, cached_mp3)
                    temp_filename = str(cached_mp3)
                except Exception as move_exc:
                    logger.warning("Could not move MP3 to cache: %s", move_exc)

            from config import OUTPUT_TTS_DIR

            # Copy final mp3 to outputs directory with timestamp for user reference
            import shutil, datetime

            out_name = (
                OUTPUT_TTS_DIR
                / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{cache_key}.mp3"
            )
            try:
                shutil.copy2(temp_filename, out_name)
                logger.debug("Saved TTS output to %s", out_name)
            except Exception as copy_exc:
                logger.warning("Failed to save TTS output: %s", copy_exc)

            logger.debug("Playing speech from file %s", temp_filename)

            try:
                # Initialize pygame mixer
                pygame.mixer.init()

                try:
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()

                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)

                    _notify(status_callback, "Text to speech completed!")
                    logger.info("Text-to-speech playback completed successfully")
                    return True
                except Exception:
                    _notify(status_callback, "Error: Audio playback failed!")
                    logger.warning(
                        "Audio file produced but playback failed; continuing without playback."
                    )
                    return True  # File was produced successfully
                finally:
                    pygame.mixer.quit()
            finally:
                # Attempt to delete the file with retries
                for attempt in range(3):
                    try:
                        os.remove(temp_filename)
                        break
                    except Exception as exc:
                        logger.debug(
                            "Attempt %d to delete temp file failed: %s",
                            attempt + 1,
                            exc,
                        )
                        if attempt == 2:
                            logger.warning(
                                "Could not delete temp file %s after 3 attempts",
                                temp_filename,
                            )
                        else:
                            time.sleep(0.1)
                return True
        except ValueError as ve:
            logger.error("Error in text_to_speech: %s", ve)
            return False
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected error in text_to_speech: %s", exc)
            return False

    # ---------------------------------------------------------------------
    # Speech -> Text
    # ---------------------------------------------------------------------

    def speech_to_text(
        self,
        language: str = "en",
        timeout: int = 5,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Capture audio from the default microphone and transcribe it."""
        try:
            logger.info("Starting speech recognition. Language: %s", language)
            _notify(status_callback, "Calibrating microphone...")

            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                _notify(status_callback, "Listening...")

                try:
                    audio = self.recognizer.listen(source, timeout=timeout)
                    _notify(status_callback, "Processing speech...")

                    # Extract audio features
                    audio_data = (
                        np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32)
                        / 32768.0
                    )
                    sample_rate = audio.sample_rate

                    if self.anomaly_detector.detect_audio_anomaly(
                        audio_data, sample_rate
                    ):
                        raise ValueError("Audio contains suspicious patterns")

                    # Convert audio to text
                    text = self.recognizer.recognize_google(audio, language=language)
                    logger.info("Recognized text: %s", text)

                    # Check for offensive content
                    if self.anomaly_detector.detect_offensive_content(text, language):
                        raise ValueError("Recognized text contains offensive content")

                    from config import OUTPUT_STT_DIR
                    import datetime, pathlib

                    try:
                        outfile = (
                            OUTPUT_STT_DIR
                            / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        )
                        pathlib.Path(outfile).write_text(text, encoding="utf-8")
                        logger.debug("Saved STT result to %s", outfile)
                    except Exception as write_exc:
                        logger.warning("Failed to write STT output: %s", write_exc)

                    return text
                except sr.WaitTimeoutError:
                    raise ValueError("No speech detected")
                except sr.UnknownValueError:
                    raise ValueError("Could not understand audio")
                except sr.RequestError as exc:
                    raise ValueError(
                        f"Speech recognition service error: {exc}"
                    ) from exc
        except Exception as exc:
            # Notify any status callback (e.g., GUI) about the specific failure reason
            _notify(status_callback, str(exc))
            logger.error("Error in speech_to_text: %s", exc)
            return ""
