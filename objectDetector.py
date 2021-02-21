import cv2
import numpy as np

threshold = 0.45 # Threshold to detect object
nms_threshold = 0.5
enterCount = 0
exitCount = 0
id_count = 0
offsetRefLines = 150
offXRef = 600

cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
person_id = []

classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def checkLineCrossing(x,coorXLine):
    absDistance = abs(x-coorXLine)
    if absDistance<=2:
        return 1
    else:
        return 0
def checkLeavingImg(x):
    if x <= 1 or x>=720:
        return 1
    else:
        return 0
def setId(idName):
    person_id.append(idName)
def removeId(idName):
    if idName in person_id:
        person_id.remove(idName)

while True:
    success,img = cap.read()
    cv2.line(img,(offXRef,0),(offXRef,720),(255,0,0),thickness=2)

    classIds, confs, bbox = net.detect(img,confThreshold=threshold)
    bbox = list(bbox) #numpy array --> list
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(classIds,bbox)

    indices = cv2.dnn.NMSBoxes(bbox,confs,threshold,nms_threshold) # non maximum suppression

    for i in indices:
        i = i[0]
        if classIds[i][0] == 1: #person
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            coorXCentroid = int((x+x+w)/2)
            coorYCentroid = int((y+y+h)/2)
            #print(coorXCentroid)
            personCentroid = (coorXCentroid,coorYCentroid)
            cv2.circle(img,personCentroid,3,color=(0,0,255),thickness=5)
            if (checkLeavingImg(coorXCentroid) and len(person_id)>0):
                removeId(i)
            if (checkLineCrossing(coorXCentroid,offXRef)):
                if i in person_id:
                    continue
                else:
                    enterCount += 1
                    setId(i)
                    print(person_id)

            cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,"Unique id: {}".format(str(i)).upper(),(box[0]+10,box[1]+60),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img, "Entrances: {}".format(str(enterCount)), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
    # if len(classIds) != 0: #have things to detect
    #     for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         if classId == 1: #person
    #             cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #             cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
    #                         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #             cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    #                         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow("Output",img)
    cv2.waitKey(1)