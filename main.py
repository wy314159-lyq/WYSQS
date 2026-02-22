# main.py
# Entry point for the SQS Generator application.

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor

from gui.main_window import MainWindow


def apply_dark_theme(app: QApplication) -> None:
    """Apply a dark color palette to the application."""
    app.setStyle("Fusion")
    palette = QPalette()

    dark_bg     = QColor(30, 30, 46)
    mid_bg      = QColor(45, 45, 65)
    light_bg    = QColor(60, 60, 80)
    text        = QColor(220, 220, 230)
    dim_text    = QColor(140, 140, 160)
    highlight   = QColor(60, 120, 200)
    bright_text = QColor(255, 255, 255)
    link        = QColor(100, 160, 255)

    palette.setColor(QPalette.Window,          dark_bg)
    palette.setColor(QPalette.WindowText,      text)
    palette.setColor(QPalette.Base,            mid_bg)
    palette.setColor(QPalette.AlternateBase,   light_bg)
    palette.setColor(QPalette.ToolTipBase,     mid_bg)
    palette.setColor(QPalette.ToolTipText,     text)
    palette.setColor(QPalette.Text,            text)
    palette.setColor(QPalette.Button,          mid_bg)
    palette.setColor(QPalette.ButtonText,      text)
    palette.setColor(QPalette.BrightText,      bright_text)
    palette.setColor(QPalette.Link,            link)
    palette.setColor(QPalette.Highlight,       highlight)
    palette.setColor(QPalette.HighlightedText, bright_text)
    palette.setColor(QPalette.Disabled, QPalette.Text,       dim_text)
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, dim_text)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, dim_text)

    app.setPalette(palette)
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #404060;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 4px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #aaaacc;
        }
        QTableWidget {
            gridline-color: #404060;
            selection-background-color: #3a5a9a;
        }
        QHeaderView::section {
            background-color: #303050;
            color: #ccccdd;
            padding: 4px;
            border: 1px solid #404060;
            font-weight: bold;
        }
        QScrollBar:vertical {
            background: #2a2a3a;
            width: 10px;
        }
        QScrollBar::handle:vertical {
            background: #505070;
            border-radius: 4px;
        }
        QTabWidget::pane {
            border: 1px solid #404060;
        }
        QTabBar::tab {
            background: #303050;
            color: #aaaacc;
            padding: 6px 16px;
            border: 1px solid #404060;
            border-bottom: none;
        }
        QTabBar::tab:selected {
            background: #1e1e2e;
            color: #ffffff;
        }
        QProgressBar {
            border: 1px solid #404060;
            border-radius: 4px;
            text-align: center;
            color: white;
        }
        QProgressBar::chunk {
            background: #2a6a2a;
            border-radius: 3px;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background: #303050;
            border: 1px solid #505070;
            border-radius: 3px;
            padding: 2px 4px;
            color: #ddddee;
        }
        QPushButton {
            background: #303050;
            border: 1px solid #505070;
            border-radius: 4px;
            padding: 4px 10px;
            color: #ddddee;
        }
        QPushButton:hover {
            background: #404060;
        }
        QPushButton:pressed {
            background: #252540;
        }
    """)


def main() -> None:
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SQS Generator")
    app.setOrganizationName("Materials Science")

    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
