#!/usr/bin/env python

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import numpy as np
import sys, os, glob, random, string, subprocess
import dlib
import win32api, win32con
import logging, time
from skimage import io
from multiprocessing import Process, Value
import threading
from datetime import datetime
from voiceRecord import *
from variance import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')

salt = ''.join(random.sample(string.digits[2:], 8))

CAMERA_FPS = 30

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

faceAllPointXQueue = VectorQueue(dim = 68, len = CAMERA_FPS)
faceAllPointYQueue = VectorQueue(dim = 68, len = CAMERA_FPS)
faceMousePointXQueue = VectorQueue(dim = 20, len = CAMERA_FPS / 2)
faceMousePointYQueue = VectorQueue(dim = 20, len = CAMERA_FPS / 2)

def draw_rectangle(frame, det, color=(100, 200, 100)):
    cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), color)

def draw_point_line(frame, det, color=(100, 230, 110)):
    shape = sp(frame, det)
    dicout = {49:59, 50:58, 51:57, 52:56, 53:55}
    dicin = {61:67, 62:66, 63:65}
    dicx = {48:54}

    faceVecX = np.zeros(68)
    faceVecY = np.zeros(68)
    mouseVecX = np.zeros(20)
    mouseVecY = np.zeros(20)

    for i in range(68):
        faceVecX[i] = shape.part(i).x
        faceVecY[i] = shape.part(i).y
        if i >= 48:
            mouseVecX[i - 48] = shape.part(i).x
            mouseVecY[i - 48] = shape.part(i).y

    faceAllPointXQueue.push(faceVecX)
    faceAllPointYQueue.push(faceVecY)

    faceMousePointXQueue.push(mouseVecX)
    faceMousePointYQueue.push(mouseVecY)

    # print(np.sum(faceAllPointXQueue.var()) / 68)
    print(np.sum(faceMousePointYQueue.var()) / 20)

    for i in range(shape.num_parts):
        cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (250, 40, 80))
    for i in dicx:
        oriu = (shape.part(i).x, shape.part(i).y)
        oriv = (shape.part(dicx[i]).x, shape.part(dicx[i]).y)
        c = (0, 0, 255)
        cv2.line(frame, oriu, oriv, c) 
    for i in dicout:
        oriu = (shape.part(i).x, shape.part(i).y)
        oriv = (shape.part(dicout[i]).x, shape.part(dicout[i]).y)
        c = (255, 0, 0)
        cv2.line(frame, oriu, oriv, c) 
    for i in dicin:
        oriu = (shape.part(i).x, shape.part(i).y)
        oriv = (shape.part(dicin[i]).x, shape.part(dicin[i]).y)
        c = (0, 255, 0)
        cv2.line(frame, oriu, oriv, c)

def lips_motion_checker(video):
    num_frames = CAMERA_FPS * 5
    # 5 seconds frames
    
    frames = []
    lefteye = []
    righteye = []

    # Start time
    start = time.time()

    for i in range(num_frames):
        ret, frame = video.read()
        if ret == True:
            frame = cv2.flip(frame,1)
            frames.append(frame)

    logger.info('get %d frames', len(frames))

    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    fps  = num_frames / seconds

    # Calculate frames per second
    logger.info("Time taken : {0} seconds".format(seconds))
    logger.info("Estimated frames per second : {0}".format(fps))

    outleft = cv2.VideoWriter('left.avi' ,cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1280,720))
    outright = cv2.VideoWriter('right.avi',cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1280,720))

    frame_cnt = 0
    faceRect = None

    # lips activity detection windows set to half second
    mouseYQueue = VectorQueue(20, CAMERA_FPS / 2)

    for f in frames:
        leftframe = f[:, :1280, :]
        outleft.write(leftframe)
        lefteye.append(leftframe)

        rightframe = f[:, 1280:, :]
        outright.write(rightframe)
        righteye.append(rightframe)
        
        
        if frame_cnt % CAMERA_FPS == 0:
            dets = detector(leftframe, 1)
            if len(dets) != 1:
                logging.error('key frame face detection failure')
                continue
            faceRect = dets[0]

        shape = sp(leftframe, faceRect)
        vec = np.zeros(20)
        for i in range(48, 68):
            vec[i - 48] = shape.part(i).y
        mouseYQueue.push(vec)
        print('var = ', np.sum(mouseYQueue.var()))
        frame_cnt += 1

    outleft.release()
    outright.release()
    logger.info('output two eyes video')

class MainApp(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.checking_state = Value('i', 0)
        self.setWindowTitle('身份识别认证')
        self.video_size = QSize(800, 480)
        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_left_label = QLabel()
        self.image_right_label = QLabel()
        self.image_left_label.setFixedSize(self.video_size)
        self.image_right_label.setFixedSize(self.video_size)
        # self.image_label.setAlignment(Qt.AlignCenter)

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


        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.image_left_label)
        self.image_layout.addWidget(self.image_right_label)

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.image_layout)
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

        # proc = Process(target=checker, args=(frame, self.checking_state, ))
        # proc = threading.Thread(target=checker, args=(frame, self.checking_state, ))
        proc = threading.Thread(target=lips_motion_checker, args=(self.capture, ))
        proc.start()

    def setup_camera(self, fps = 30):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
        height = frame.shape[0]
        weight = frame.shape[1]
        # dets = detector(frame, 1)
        # dets = []
        # for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0))

        frame_left = frame[:, :1280, :]
        # cv2.rectangle(frame_left, (390, 610), (890, 110), (0, 255, 0))
        frame_right = frame[:, 1280:, :]
        # cv2.rectangle(frame_right, (390, 610), (890, 110), (0, 255, 0))
        
        # image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        frame_left = cv2.resize(frame_left, (800, 480))
        frame_right = cv2.resize(frame_right, (800, 480))

        dets_left = dlib.rectangle(left=250,right=550,top=90,bottom=390)
        # dets = detector(frame_left, 1)
        # for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            # print("Hei : {}, Wei : {}".format(d.top() - d.bottom(), d.right() - d.left()))
            # cv2.rectangle(frame_left, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 200))
            # newrec = dlib.rectangle(left=d.left()-43,bottom=d.bottom()-53,right=d.right()-40,top=d.top()-20)
            # draw_rectangle(frame_left, newrec, (0, 255, 200))
            # draw_point_line(frame_left, newrec)
        draw_rectangle(frame_left, dets_left)
        draw_point_line(frame_left, dets_left)

        image_left = QImage(frame_left.tobytes(), frame_left.shape[1], frame_left.shape[0], QImage.Format_RGB888)
        image_right = QImage(frame_right.tobytes(), frame_right.shape[1], frame_right.shape[0], QImage.Format_RGB888)

        if self.checking_state.value == 0:
            # if len(dets) == 1:
            self.text_label.setText('等待识别')
            self.check_button.setEnabled(True)
            # else:
                # self.text_label.setText('未检测到唯一人脸')
                # self.check_button.setDisabled(True)
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
        self.image_left_label.setPixmap(QPixmap.fromImage(image_left))
        self.image_right_label.setPixmap(QPixmap.fromImage(image_right))

if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
