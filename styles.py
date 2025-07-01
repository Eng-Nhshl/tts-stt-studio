"""Centralized Qt stylesheet strings for custom widgets."""

GLASS_FRAME = """
GlassFrame {
    background: rgba(255,255,255,0.15);
    border-radius: 24px;
    border: 1.5px solid rgba(255,255,255,0.22);
}
"""

NEON_BUTTON = """
QPushButton {
    color: #fff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d1cbe, stop:1 #00d2ff);
    border-radius: 18px;
    border: none;
    padding: 10px 28px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:1, y1:0, x2:0, y2:1, stop:0 #00d2ff, stop:1 #3d1cbe);
}
QPushButton:pressed {
    background: #0c024d;
}
"""

TITLE_LABEL = """
QLabel {
    font-size: 20px;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    color: #fff;
    margin: 18px 0 0 0;
    font-weight: 600;
    letter-spacing: 1.5px;
    background: qlineargradient(x1:0 y1:0, x2:0 y2:1, stop:0 rgba(0,210,255,0.3), stop:1 rgba(61,28,190,0.3));
}
"""

GLASS_TAB_WIDGET = """
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
    font-size: 15px;
    margin-right: 4px;
    padding: 8px 24px;
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3d1cbe, stop:1 #00d2ff);
    color: #fff;
}
"""

GLASS_COMBO_BOX = """
QComboBox {
    background: rgba(255,255,255,0.18);
    color: #fff;
    border-radius: 12px;
    border: 1px solid #3d1cbe;
    padding: 6px 18px;
    font-size: 15px;
}
QComboBox QAbstractItemView {
    background: #24243e;
    color: #fff;
}
"""

GLASS_TEXT_EDIT = """
QTextEdit {
    background: rgba(255,255,255,0.10);
    color: #fff;
    border-radius: 12px;
    border: 1.5px solid #00d2ff;
    font-size: 17px;
    padding: 8px 10px;
    min-height: 80px;
}
"""

GLASS_LABEL = """
QLabel {
    color: #e0e5ff;
    font-size: 14px;
    margin-top: 12px;
    font-weight: 500;
    background: qlineargradient(x1:0 y1:0, x2:0 y2:1, stop:0 rgba(224,229,255,0.2), stop:1 rgba(224,229,255,0.1));
}
"""

__all__ = [
    "GLASS_FRAME",
    "NEON_BUTTON",
    "TITLE_LABEL",
    "GLASS_TAB_WIDGET",
    "GLASS_COMBO_BOX",
    "GLASS_TEXT_EDIT",
    "GLASS_LABEL",
]
