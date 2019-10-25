import dlib
import numpy
import cv2
import os
import csv
import pickle
import math

#input directory
data_directory='input'
dirlist=os.listdir(data_directory)
label_dir=os.listdir(data_directory)
for i in range(len(dirlist)) :
    dirlist[i]=data_directory+'/'+dirlist[i]
print(dirlist)


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

#calculates square of distances btw points
def dist(x,y):
    return math.sqrt(((x[0,0]-y[0,0])**2+(x[0,1]-y[0,1])**2))



#produces individual feature vector
def ele(im):
    points = get_landmarks(im)
#right eye points
    er=points[36:42]
#left eye points
    el=points[42:48]
    x=0
    y=0
    # centroid of right eye
    for i in range(6) :
       x+=er[i,0]
       y+=er[i,1]
    cerx=x/6
    cery=y/6

# centroid of left eye
    x=0
    y=0
    for i in range(6) :
       x+=el[i,0]
       y+=el[i,1]
    celx=x/6
    cely=y/6

    ox=(cerx+celx)/2
    oy=(cery+cely)/2

    #DISTANCE SQUARE btw two eyes
    d2=math.sqrt((cerx-celx)**2+(cery-cely)**2)


    feature=[]
# dividing d2 to normalize
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




for i in range(len(dirlist)):
    for images in os.listdir(dirlist[i]):
        t=dirlist[i]+'/'+images
        im = cv2.imread(t)
        print(dirlist[i])
        print('processing ' + images)
        temp=ele(im)
        print(temp)
        features.append(temp)
        labels.append(label_dir[i])

#producing the input for model_train file
pickle.dump( features, open( "output_features.p", "wb" ) )
pickle.dump( labels, open( "output_labels.p", "wb" ) )
