import dlib
import numpy
import cv2
import os
import pickle
import math
from sklearn import svm

clf=pickle.load(open("class.p","rb"))



features=[]
labels=[]


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def dist(x,y):
    return math.sqrt(((x[0,0]-y[0,0])**2+(x[0,1]-y[0,1])**2))




def ele(im):
    points = get_landmarks(im)


    er=points[36:42]
    el=points[42:48]
    x=0
    y=0
    for i in range(6) :
       x+=er[i,0]
       y+=er[i,1]
    cerx=x/6
    cery=y/6


    x=0
    y=0
    for i in range(6) :
       x+=el[i,0]
       y+=el[i,1]
    celx=x/6
    cely=y/6

    ox=(cerx+celx)/2
    oy=(cery+cely)/2
    #DISTANCE SQUARE
    d2=math.sqrt((cerx-celx)**2+(cery-cely)**2)


    feature=[]

    t=dist(points[27],points[0])/(d2)
    feature.append(t)
    t=dist(points[27],points[1])/(d2)
    feature.append(t)
    t=dist(points[27],points[2])/(d2)
    feature.append(t)
    t=dist(points[27],points[3])/(d2)
    feature.append(t)
    t=dist(points[27],points[4])/(d2)
    feature.append(t)
    t=dist(points[27],points[5])/(d2)
    feature.append(t)
    t=dist(points[27],points[6])/(d2)
    feature.append(t)
    t=dist(points[27],points[7])/(d2)
    feature.append(t)
    t=dist(points[27],points[8])/(d2)
    feature.append(t)
    t=dist(points[27],points[9])/(d2)
    feature.append(t)
    t=dist(points[27],points[10])/(d2)
    feature.append(t)
    t=dist(points[27],points[11])/(d2)
    feature.append(t)
    t=dist(points[27],points[12])/(d2)
    feature.append(t)
    t=dist(points[27],points[13])/(d2)
    feature.append(t)
    t=dist(points[27],points[14])/(d2)
    feature.append(t)
    t=dist(points[27],points[15])/(d2)
    feature.append(t)
    t=dist(points[27],points[16])/(d2)
    feature.append(t)
    t=dist(points[27],points[17])/(d2)
    feature.append(t)
    t=dist(points[39],points[42])/(d2)
    feature.append(t)
    t=dist(points[51],points[27])/(d2)
    feature.append(t)
    return (feature)


dir_test='/home/test'

for images in os.listdir(dir_test):
    t=dir_test+'/'+images
    im = cv2.imread(t)

    print('processing ' + images)
    temp=ele(im)
    features.append(temp)


ans=(clf.predict(features))
i=0
for images in os.listdir(dir_test):

    print(images +'  :  ' +ans[i])
    i+=1
