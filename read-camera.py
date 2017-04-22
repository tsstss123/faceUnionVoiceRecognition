#!/usr/bin/env python

#from PySide.QtCore import *
#from PySide.QtGui import *
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

TIME = 10
NUM_SAMPLES = 2000  
framerate = 16000  
channels = 1  
sampwidth = 2 

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

salt = ''.join(random.sample(string.digits, 8))


def change_salt():
    global salt
    salt = ''.join(random.sample(string.digits, 8))

def save_wave_file(filename, data):
    '''save the date to the wav file'''
    framerate = 16000
    channels = 1
    sampwidth = 2
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def record_wave(filename=""):
    #open the input of wave
    TIME = 10
    NUM_SAMPLES = 2000
    pa = PyAudio()
    stream = pa.open(format = paInt16, channels = 1,
                    rate = framerate, input = True, 
                    frames_per_buffer = NUM_SAMPLES)
    save_buffer = []
    count = 0
    while count < TIME * 8:
        #read NUM_SAMPLES sampling data
        string_audio_data = stream.read(NUM_SAMPLES)
        save_buffer.append(string_audio_data)
        count += 1
        print('recording...')

    # filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+".wav"
    if filename == "":
        filename = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    filename += ".wav"
    # save_wave_file(r'D:\goWorkSpacce\src\github.com\liuxp0827\govpr\example' + '\\' + filename, save_buffer)
    save_wave_file('wav\\' + filename, save_buffer)
    save_buffer = []
    return filename

people = []
def load_db(db_name = 'vector.txt'):
    global people
    with open(db_name, 'r') as f:
        vector_list = f.readlines()
    for x in vector_list:
        x = x.strip()
        people.append((np.array(x.split()[:-1]).astype(np.float64), x.split()[-1]))
    print('load ' + str(len(people)) +' faces')

def checker(frame, state):
    state.value = 1
    global people
    audio_filename = record_wave()
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
        print(type(face_descriptor))
        d_test = np.array(face_descriptor).astype(np.float64)
    state.value = 3
    maxlike = 1e8
    pname = 'MATCH FAIL'
    print(len(people), 'need to match')
    for p in people:
        D = np.linalg.norm(p[0] - d_test)
        # print('D to ' + p[1] + ' = ' + str(D))
        if D < maxlike:
            pname = p[1]
            maxlike = D
    os.chdir(r'wav')
    train_file = r'train\01_32468975.wav'
    args = [
        r'vpr.exe',
        train_file,
        os.path.abspath(audio_filename)
    ]
    audio_p = subprocess.Popen(args, stdout=subprocess.PIPE)
    audio_p.wait()
    audio_tho = audio_p.stdout.readlines()
    audio_tho = float(audio_tho[0][:-1])
    pass_flag = False
    if maxlike < 0.4:
        print(pname)
        pass_flag = True
    else:
        print('no people in library')
        pass_flag = False
    if pass_flag:
        title = "认证通过"
        pname = pname.split('_')[0]
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
        self.check_button = QPushButton("Check Face")
        self.check_button.setFont(button_font)
        self.check_button.setDisabled(True)
        self.check_button.clicked.connect(self.check_face)

        self.quit_button = QPushButton("Quit")
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
        elif self.checking_state.value == 2:
            self.text_label.setText('提取特征中')
        elif self.checking_state.value == 3:
            self.text_label.setText('识别中')
        elif self.checking_state.value == 4:
            self.checking_state.value = 0
            change_salt()

        self.salt_label.setText(salt)
        self.image_label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
