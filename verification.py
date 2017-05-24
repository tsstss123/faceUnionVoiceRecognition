#!/usr/bin/env python

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import sys, os, glob, random, string, subprocess
import dlib
import win32api, win32con
from skimage import io
from multiprocessing import Process, Value
from datetime import datetime
from voiceRecord import *

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

salt = ''.join(random.sample(string.digits[2:], 8))

def change_salt():
    global salt
    salt = ''.join(random.sample(string.digits[2:], 8))

people = []

def load_db():
    global people
    peoplelist_path = 'dataset/peoplelist.txt'
    with open(peoplelist_path, 'r') as f:
        namelist = f.readlines()
    for name in namelist:
        name = name.strip()
        with open('dataset/' + name + '/vector.txt', 'r') as f:
            vector_list = f.readlines()
        for x in vector_list:
            x = x.strip()
            people.append((np.array(x.split()[:-1]).astype(np.float64), x.split()[-1]))
    print('load ' + str(len(people)) +' vectors')

def checker(frame, state):
    state.value = 1
    global people
    audio_filename = record_wave(prefix='dataset/tmp/')
    print(audio_filename, 'saved')
    load_db()
    state.value = 2
    dets = detector(frame, 1)
    print('detect ' + str(len(dets)) + ' faces')
    if len(dets) > 1 or len(dets) == 0:
        print('fail')
        return
    for k, d in enumerate(dets):
        shape = sp(frame, d)
        face_descriptor = facerec.compute_face_descriptor(frame, shape, 10)
        #print(type(face_descriptor))
        d_test = np.array(face_descriptor).astype(np.float64)
    state.value = 3
    maxlike = 1e8
    audio_tho = random.uniform(0,0.5)
    pname = 'MATCH FAIL'
    print(len(people), 'need to match')
    for p in people:
        D = np.linalg.norm(p[0] - d_test)
        if D < maxlike:
            pname = p[1]
            maxlike = D

    pass_flag = False
    
    if maxlike < 0.4:
        os.chdir('dataset')
        args = ['vpr.exe']
        for fname in os.listdir(pname):
            if '.wav' in fname:
                args.append(os.path.abspath(pname + '/' + fname))
                # print('train file ', os.path.abspath(pname + '/' + fname))
        args.append('tmp/' + audio_filename)
        # print('test flie ', 'tmp/' + audio_filename)
        audio_p = subprocess.Popen(args, stdout=subprocess.PIPE)
        audio_p.wait()
        audio_tho = audio_p.stdout.readlines()
        audio_tho = float(audio_tho[0][:-1])
        if audio_tho > 1:
            pass_flag = True
    else:
        print('no people in library')
        pass_flag = False
    
    if pass_flag:
        title = "认证通过"
        pname = pname
    else:
        title = "认证失败"
        pname = "认证失败"
    pname += '\naudio Confidence = ' + str(round(audio_tho,4))
    pname += '\nimage Confidence = ' + str(round(maxlike, 4))
    state.value = 4
    win32api.MessageBox(0, pname, title, win32con.MB_OK)


class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.checking_state = Value('i', 0)
        self.setWindowTitle('身份识别认证')
        self.video_size = QSize(640, 480)
        self.setup_ui()
        self.setup_camera()

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
        self.check_button = QPushButton("认证")
        self.check_button.setFont(button_font)
        self.check_button.setDisabled(True)
        self.check_button.clicked.connect(self.check_face)

        self.quit_button = QPushButton("退出")
        self.quit_button.setFont(button_font)
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.text_label)
        self.main_layout.addWidget(self.salt_label)
        self.main_layout.addWidget(self.check_button)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

    def check_face(self):

        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        change_salt()
        
        proc = Process(target=checker, args=(frame, self.checking_state, ))
        proc.start()

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
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0))
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)

        if self.checking_state.value == 0:
            if len(dets) == 1:
                self.text_label.setText('可以进行识别')
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
            self.text_label.setText('识别中')
            self.check_button.setDisabled(True)
        elif self.checking_state.value == 4:
            self.checking_state.value = 0
            self.check_button.setEnabled(True)
            change_salt()

        self.salt_label.setText(salt)
        self.image_label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
