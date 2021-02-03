
import cv2 as cv
import numpy as np
 
cap = cv.VideoCapture(0) #Webcam number
whT = 320 #Width,Height of Target,since both width and height are same(squared image) we wrote it as a single parameter
confThreshold =0.5 #confidence threshold for taking note of acceptable probability thresholds
nmsThreshold= 0.2 #Suppression function threshold,takes the threshold for suppressing the duplicate bounding boxes
 
#### LOAD MODEL
## Coco Names
classesFile = "/home/madhavi/coding/YOLO_v3/coco.names" #This file has all the names of classes.Here it has 80 classes in COCO dataset
classNames = [] #make an empty list for classes
with open(classesFile, 'rt') as f: #rt is read-text
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

## Model Files
modelConfiguration = "/home/madhavi/coding/YOLO_v3/yolov3-320.cfg" #Yolo Configuration file for 320*320 resolution
modelWeights = "/home/madhavi/coding/YOLO_v3/yolov3.weights" #Yolo Weights file for 320*320 resolution
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #Reading configuration and weights file from Darknet(It is written in C and specialised for object detection,works like backend NN,very efficient than other NN)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) #Setting preferences,backend is run on OpenCV
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) #Setting preferences,target host is CPU,for GPU use CUDA
 
def findObjects(outputs,img):
    hT, wT, cT = img.shape #height,width and channel
    bbox = [] #contains list of all bounding boxes
    classIds = [] #contains list of all class IDs
    confs = [] #confidence values
    for output in outputs:
        for det in output: #output gives an array with (300,85) shape or any other shape x based on the no.of pics and y is 85 = 80 COCO classes + height + width + center-x + center-y
            scores = det[5:] #ignoring the 5 values,I need to check if any class has a good detection probability,so I took scores starting from 5 to 85 
            classId = np.argmax(scores) #argmax gives the index of the highest value,i.e index of class ID
            confidence = scores[classId] #value of that max. value is confidence
            if confidence > confThreshold: #if the confidence is greater than threshold
                w,h = int(det[2]*wT) , int(det[3]*hT) #det[2] and det[3] are not pixel values to get them we need to multiply with pixel values
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h]) #appending those values to the bounding box list 
                classIds.append(classId) #appending corresponding class
                confs.append(float(confidence)) #appending corresponding confidence
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold) #Maximum Suppression function,used to eliminate duplicate bounding boxes
 
    for i in indices:
        i = i[0] #values in indices are in form of nested list,so we are removing redundant list.
        box = bbox[i] #we are creating a box list to take the values for x,y,w,h
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2) #Drawing the rectangle,parameters are image, (x,y), corner point(x+w,y+h), RGB value,thickness
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
 
 
while True:
    success, img = cap.read() #returns a tuple(successful,image).If successful,then image is read into img
 
    #Blob is Binary Large Objects,it is used to store data in binary formats and is used for multimedia files.For sending to opencv we need images in blob format.
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False) #scalefactor=1/255, size specified using whT, mean ,no cropping(i.e no cutting black portions generated during rescaling like from landscape to potrait)
    net.setInput(blob) #setting blob as input
    layersNames = net.getLayerNames() #There are various layers in our NN architecture,so we want to get the layernames,layernames are giving us indices satrting at 1 and not 0
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()] #So we took layername index, subtracted 1 from it to get output names
    outputs = net.forward(outputNames) #We are interested in outputs of only these three outputnames.Output names are at yolo_82,yolo_94,yolo_106.These are important as yolo detects objects at these three layers only and we get three bounding boxes corresponding to these three outputs
    findObjects(outputs,img)
 
    cv.imshow('Image', img)
    cv.waitKey(1) #waitKey is nothing but delay,here it delays for 1ms
