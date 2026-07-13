
from PyQt5.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from pathlib import Path

class ProgressBar(QDialog):
    def __init__(self, parent=None):
        super(ProgressBar, self).__init__(parent)

        self.setFixedSize(440, 160)
        self.setWindowTitle(self.tr("🎬 视频保存中..."))
        icon_path = Path(__file__).with_name("1689818874703513.png")
        self.setWindowIcon(QIcon(str(icon_path)))

        self.setStyleSheet("""
            QDialog {
                background-color: #FAFAFC;
            }
            QLabel {
                font-family: 'Verdana';
                font-size: 12pt;
                color: #002B36;
            }
            QProgressBar {
                border: 1px solid #AAAAAA;
                border-radius: 6px;
                background: #E8E8E8;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3AA1FF;
                width: 20px;
            }
            QPushButton {
                font-size: 12pt;
                color: white;
                background-color: #FF6666;
                border: none;
                padding: 6px 14px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #FF3333;
            }
        """)

        # 横线装饰条
        self.top_line = QFrame(self)
        self.top_line.setFrameShape(QFrame.HLine)
        self.top_line.setStyleSheet("background-color: #3AA1FF;")
        self.top_line.setFixedHeight(3)

        self.TipLabel = QLabel(self.tr("当前帧 / 总帧数：0 / 0"))
        self.FeatLabel = QLabel(self.tr("保存进度："))

        self.FeatProgressBar = QProgressBar(self)
        self.FeatProgressBar.setMinimum(0)
        self.FeatProgressBar.setMaximum(100)
        self.FeatProgressBar.setValue(0)

        TipLayout = QHBoxLayout()
        TipLayout.addWidget(self.TipLabel)

        FeatLayout = QHBoxLayout()
        FeatLayout.addWidget(self.FeatLabel)
        FeatLayout.addWidget(self.FeatProgressBar)

        self.cancelButton = QPushButton('取消保存', self)

        buttonlayout = QHBoxLayout()
        buttonlayout.addStretch(1)
        buttonlayout.addWidget(self.cancelButton)

        self.footerLabel = QLabel("@LJR 所有操作将保存至本地磁盘。")
        self.footerLabel.setAlignment(Qt.AlignCenter)
        self.footerLabel.setStyleSheet("color: #888888; font-size: 9pt;")

        layout = QVBoxLayout()
        layout.addWidget(self.top_line)
        layout.addLayout(FeatLayout)
        layout.addLayout(TipLayout)
        layout.addLayout(buttonlayout)
        layout.addWidget(self.footerLabel)
        self.setLayout(layout)

        self.cancelButton.clicked.connect(self.onCancel)

    def setValue(self, start, end, progress):
        self.TipLabel.setText(self.tr(f"当前帧 / 总帧数：  {start} / {end}"))
        self.FeatProgressBar.setValue(progress)

    def onCancel(self, event=None):
        self.close()
