import sys
import numpy as np
import os.path
import cv2
import joblib

from sklearn.svm import LinearSVC

detector = cv2.FaceDetectorYN.create( "topics/NhanDangKhuonMatFacebook/face_detection_yunet_2022mar.onnx","", (320, 320), 0.9,0.3,5000)
detector.setInputSize((320, 320))
recognizer = cv2.FaceRecognizerSF.create("topics/NhanDangKhuonMatFacebook/face_recognition_sface_2021dec.onnx","")

svc = joblib.load('topics/NhanDangKhuonMatFacebook/svc.pkl')
mydict = ['BanNinh','BanThanh','ThayDuc']

def onRecognition(image):
    global imgain
    
    imgin = cv2.imread(image,cv2.IMREAD_COLOR)

    faces = detector.detect(imgin)
    face_align = recognizer.alignCrop(imgin, faces[1][0])
    face_feature = recognizer.feature(face_align)
    test_prediction = svc.predict(face_feature)

    result = mydict[test_prediction[0]]
    return result
