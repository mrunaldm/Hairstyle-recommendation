import cv2
import dlib
import numpy
import math
import pathlib
import os.path
import csv

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):
    rects = cascade.detectMultiScale(im,2.0,5)
    x,y,w,h =rects[0]
    rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def getImageName(images,directoryName):
    imageList = []
    for image in images:
        imageList.append(directoryName+"/"+image)
    return imageList
def get_lum(image, x, y, w, h, k, gray):
    if gray == 1: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    i1 = range(int(-w / 2), int(w / 2))
    j1 = range(0, h)

    lumar = numpy.zeros((len(i1), len(j1)))
    for i in i1:
        for j in j1:
            lum = numpy.min(image[y + k * h, x + i])
            lumar[i][j] = lum

    return numpy.min(lumar)

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def getEuclidianDistance(landmarks,i1,i2):
    x1 = landmarks[int(i1)][0]
    y1 = landmarks[int(i1)][1]
    x2 = landmarks[int(i2)][0]
    y2 = landmarks[int(i2)][1]

    x_co_ordinates = (x2-x1)**2
    y_co_ordinates = (y2-y1)**2
    result = math.sqrt((x_co_ordinates+y_co_ordinates))
    return result


def getAngularDistance(landmarks, i1, i2):
    x1 = landmarks[int(i1)][0]
    y1 = landmarks[int(i1)][1]
    x2 = landmarks[int(i2)][0]
    y2 = landmarks[int(i2)][1]

    x_diff = float(x1 - x2)

    if (y1 == y2): y_diff = 0.1
    if (y1 < y2): y_diff = float(numpy.absolute(y1 - y2))
    if (y1 > y2):
        y_diff = 0.1
        print("Error: Facial feature located below chin.")
    return numpy.absolute(math.atan(x_diff / y_diff))

def getHairline(NoseTip,im):
    x = NoseTip[0]
    y = NoseTip[1]
    diff = get_lum(im, x, y, 6, 2, -1, 0)
    limit = diff - 55
    while (diff > limit):
        y = int(y - 1)
        diff = get_lum(im, x, y, 6, 2, -1, 0)
    hairLine = ((x, y))
    return hairLine

def getImageValues(im):

    landmarks = get_landmarks(im)
    lmarks = landmarks.tolist()
    tip_Of_Nose = (landmarks[27][0,0],landmarks[27][0,1])
    hairLine = getHairline(tip_Of_Nose,im)
    lmarks.append(hairLine)
    cv2.circle(im, hairLine, 3, color=(255, 153, 0))
    cv2.imshow('Result',annotate_landmarks(im,get_landmarks(im)))
    return lmarks


BaseDirectory = "/home/mrunal/Desktop/Biometrics/Shapes"
shapeMapper = {"Heart":1,"Oblong":2,"Oval":3,"Round":4,"Square":5}
featureList = []
headers = ["height_to_width","jawline_to_width","chinwidth_to_width"]
for w in range(3,26):
    headers.append("Features_"+str(w))
headers.append("Shape_Mapper")
shapeDirectory = [q for q in pathlib.Path(BaseDirectory).iterdir()]
with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows([headers])
    for shape in shapeDirectory:
        image_dir = BaseDirectory+"/"+shape.name
        images = os.listdir(image_dir)
        imageList = getImageName(images,image_dir)
        for i in imageList:
            features = []
            im = cv2.imread(i)
            print i
            lmarks = getImageValues(im)
            faceWidth = getEuclidianDistance(lmarks,0,16)
            faceHeight = getEuclidianDistance(lmarks,8,68)
            jawline = getEuclidianDistance(lmarks,4,12)
            chinWidth = getEuclidianDistance(lmarks,57,8)
            features.append(faceHeight/faceWidth)
            features.append(jawline/faceWidth)
            features.append(chinWidth/faceWidth)
            for fi in range(1,8):
                distance = getEuclidianDistance(lmarks,fi,16-fi)
                features.append(distance/faceWidth)

            for fi in range(0,17):
                if fi!=8:
                    fi = getAngularDistance(lmarks,fi,8)
                    features.append(fi)

            features.append(shapeMapper.get(shape.name))
            writer.writerows([features])


cv2.destroyAllWindows()


# #This is using the Dlib Face Detector . Better result more time taking
# def get_landmarks(im):
#     rects = detector(im, 1)
#     rect=rects[0]
#     print type(rect.width())
#     fwd=int(rect.width())
#     if len(rects) == 0:
#         return None,None

#     return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]),fwd

