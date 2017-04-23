#!/usr/bin/env python

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import sys, os, glob, random, string, subprocess, wave
import dlib
import win32api, win32con
from skimage import io
from multiprocessing import Process, Value
from pyaudio import PyAudio,paInt16
from datetime import datetime
from voiceRecord import *

salt = ''.join(random.sample(string.digits[2:], 8))

def change_salt():
    global salt
    salt = ''.join(random.sample(string.digits[2:], 8))

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

def recorder(frame, state, name):
    path = 'dataset/' + name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)
    state.value = 1
    audio_filename = record_wave(prefix=path)
    print(audio_filename, 'saved')
    state.value = 2
    dets = detector(frame, 1)
    print('detect ' + str(len(dets)) + ' faces')
    if len(dets) > 1 or len(dets) == 0:
        print('fail')
        return
    for k, d in enumerate(dets):
        shape = sp(frame, d)
        face_descriptor = facerec.compute_face_descriptor(frame, shape, 20)
        print(type(face_descriptor))
        d_test = np.array(face_descriptor).astype(np.float64)
    state.value = 3
    with open(path + 'vector.txt', 'a') as fp:
        for v in face_descriptor:
            fp.write(str(v) + ' ')
        fp.write(name + '\n')
    print('output vector')
    state.value = 4

class MainApp(QWidget):
    
    def __init__(self):
        QWidget.__init__(self)
        self.get_identify()
        self.setWindowTitle(self.identify)
        self.checking_state = Value('i', 0)
        self.video_size = QSize(640, 480)
        self.setup_ui()
        self.setup_camera()
        self.read_all_people()

    def read_all_people(self):
        self.people = []
        self.people_path = 'dataset/peoplelist.txt'
        with open(self.people_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.people.append(line)
                print('dataset has', line)

    def get_identify(self):
        font = QFont()
        font.setPointSize(22)
        dlg = QInputDialog(self)
        dlg.setFont(font)
        dlg.setInputMode(QInputDialog.TextInput)
        dlg.setWindowTitle('特征录入')
        dlg.setLabelText("请输入身份标识(例如学号/姓名)")
        dlg.resize(500, 100)
        dlg.move(QDesktopWidget().availableGeometry().center() - self.frameGeometry().center())
        ok = dlg.exec_()
        self.identify = dlg.textValue()

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)
        self.image_label.setAlignment(Qt.AlignCenter)

        text_label_font = QFont()
        text_label_font.setBold(True)
        text_label_font.setPointSize(40)
        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setText('摄像头开启中...')
        self.text_label.setFont(text_label_font)

        self.salt_label = QLabel()
        self.salt_label.setAlignment(Qt.AlignCenter)
        self.salt_label.setText('动态口令载入中...')
        self.salt_label.setFont(text_label_font)

        button_font = QFont()
        button_font.setPointSize(18)
        self.check_button = QPushButton("录入特征")
        self.check_button.setFont(button_font)
        self.check_button.setDisabled(True)
        self.check_button.clicked.connect(self.check_face)

        self.write_button = QPushButton("写入特征数据库")
        self.write_button.setFont(button_font)
        self.write_button.setDisabled(True)
        self.write_button.clicked.connect(self.write_face)

        self.quit_button = QPushButton("离开")
        self.quit_button.setFont(button_font)
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.text_label)
        self.main_layout.addWidget(self.salt_label)
        self.main_layout.addWidget(self.check_button)
        self.main_layout.addWidget(self.write_button)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)
        self.center()

    def center(self):
        """make windows in desktop center
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def check_face(self):
        """Create a new process to check face
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        proc = Process(target=recorder, args=(frame, self.checking_state, self.identify, ))
        proc.start()
    
    def write_face(self):
        """Write records to file
        """
        if self.identify not in self.people:
            self.people.append(self.identify)
            with open(self.people_path, 'a') as f:
                f.write(self.identify + '\n')

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.checking_state.value = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(200)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        dets = detector(frame, 1)
        for k, d in enumerate(dets):
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0))
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)

        if self.checking_state.value == 0:
            if len(dets) == 1:
                self.text_label.setText('可以进行身份录入')
                self.check_button.setEnabled(True)
            else:
                self.text_label.setText('未检测到唯一人脸')
                self.check_button.setDisabled(True)
        elif self.checking_state.value == 1:
            self.text_label.setText('录音中')
            self.check_button.setDisabled(True)
        elif self.checking_state.value == 2:
            self.text_label.setText('提取特征中')
            self.check_button.setDisabled(True)
        elif self.checking_state.value == 3:
            self.text_label.setText('保存中')
            self.check_button.setDisabled(True)
        elif self.checking_state.value == 4:
            self.checking_state.value = 0
            self.check_button.setEnabled(True)
            self.write_button.setEnabled(True)
            change_salt()

        self.salt_label.setText(salt)
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.center()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())