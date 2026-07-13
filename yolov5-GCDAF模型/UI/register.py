
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(700, 520)
        MainWindow.setMinimumSize(QtCore.QSize(700, 520))
        MainWindow.setMaximumSize(QtCore.QSize(700, 520))
        MainWindow.setStyleSheet("QWidget {font-family: 'Verdana'; color: #202020; background: #F0FAFF;}")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 20, 700, 60))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(24)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet("color: #003366;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 70, 700, 40))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: #005580;")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.led_workerid = QtWidgets.QLineEdit(self.centralwidget)
        self.led_workerid.setGeometry(QtCore.QRect(230, 130, 240, 40))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.led_workerid.setFont(font)
        self.led_workerid.setStyleSheet("background-color: #FFFFFF; border: 1px solid #AAAAAA; border-radius: 6px; padding-left: 10px;")
        self.led_workerid.setObjectName("led_workerid")

        self.led_pwd = QtWidgets.QLineEdit(self.centralwidget)
        self.led_pwd.setGeometry(QtCore.QRect(230, 190, 240, 40))
        self.led_pwd.setFont(font)
        self.led_pwd.setStyleSheet("background-color: #FFFFFF; border: 1px solid #AAAAAA; border-radius: 6px; padding-left: 10px;")
        self.led_pwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.led_pwd.setObjectName("led_pwd")

        self.led_pwd_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.led_pwd_2.setGeometry(QtCore.QRect(230, 250, 240, 40))
        self.led_pwd_2.setFont(font)
        self.led_pwd_2.setStyleSheet("background-color: #FFFFFF; border: 1px solid #AAAAAA; border-radius: 6px; padding-left: 10px;")
        self.led_pwd_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.led_pwd_2.setObjectName("led_pwd_2")

        self.btn_reg = QtWidgets.QPushButton(self.centralwidget)
        self.btn_reg.setGeometry(QtCore.QRect(240, 320, 220, 42))
        font_btn = QtGui.QFont()
        font_btn.setPointSize(14)
        self.btn_reg.setFont(font_btn)
        self.btn_reg.setStyleSheet("background-color: #3399FF; color: white; border-radius: 20px;")
        self.btn_reg.setObjectName("btn_reg")

        self.btn_login = QtWidgets.QPushButton(self.centralwidget)
        self.btn_login.setGeometry(QtCore.QRect(270, 390, 160, 30))
        font_login = QtGui.QFont()
        font_login.setPointSize(12)
        self.btn_login.setFont(font_login)
        self.btn_login.setStyleSheet("QPushButton { color: #0066CC; background-color: transparent; border: none; } QPushButton:hover { text-decoration: underline; }")
        self.btn_login.setObjectName("btn_login")

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.label_footer = QtWidgets.QLabel(self.centralwidget)
        self.label_footer.setGeometry(QtCore.QRect(0, 480, 700, 30))
        self.label_footer.setText("欢迎使用")
        self.label_footer.setAlignment(QtCore.Qt.AlignCenter)
        self.label_footer.setStyleSheet("color: gray; font-size: 10pt; font-family: Verdana;")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "注册"))
        self.label.setText(_translate("MainWindow", "人脸口罩佩戴检测系统"))
        self.label_2.setText(_translate("MainWindow", "注册新账号"))
        self.led_workerid.setPlaceholderText(_translate("MainWindow", "请输入用户名"))
        self.led_pwd.setPlaceholderText(_translate("MainWindow", "请输入密码"))
        self.led_pwd_2.setPlaceholderText(_translate("MainWindow", "请确认密码"))
        self.btn_reg.setText(_translate("MainWindow", "注册"))
        self.btn_login.setText(_translate("MainWindow", "已有账号，返回登录"))
