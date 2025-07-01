from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QPushButton,
    QLabel,
    QTabWidget,
    QComboBox,
    QTextEdit,
)

import styles as _styles

__all__ = [
    "GlassFrame",
    "NeonButton",
    "TitleLabel",
    "GlassTabWidget",
    "GlassComboBox",
    "GlassTextEdit",
    "GlassLabel",
]


class GlassFrame(QFrame):
    """A semi-transparent frosted-glass effect frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_styles.GLASS_FRAME)


class NeonButton(QPushButton):
    """A neon-style push button."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(_styles.NEON_BUTTON)


class TitleLabel(QLabel):
    """Glowing title label."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setStyleSheet(_styles.TITLE_LABEL)


class GlassTabWidget(QTabWidget):
    """Tab widget with glass aesthetics."""

    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet(_styles.GLASS_TAB_WIDGET)


class GlassComboBox(QComboBox):
    """Combo box with transparent backdrop matching the glass theme."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(_styles.GLASS_COMBO_BOX)


class GlassTextEdit(QTextEdit):
    """Text area styled for dark glass UI."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet(_styles.GLASS_TEXT_EDIT)


class GlassLabel(QLabel):
    """Secondary label for hint/status texts inside glass panels."""

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setStyleSheet(_styles.GLASS_LABEL)
