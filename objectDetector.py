import cv2
import numpy as np
import time

threshold = 0.45 # Threshold to detect object
nms_threshold = 0.5
enterCount = 0
exitCount = 0
id_count = 0
offsetRefLines = 150
offXRef = 600

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
cap.set(5,30)

classNames= []
person_dict = dict()
tempCoor = dict()

boundary1 = 480
boundary2 = 720
ex_boundary1 = 240
ex_boundary2 = 960

prev_frame = 0
new_frame = 0

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
def checkZoneCrossing(x,y,direction):
    bound1Distance = abs(x-boundary1)
    bound2Distance = abs(x-boundary2)
    #print("bound1 dis: "+str(bound1Distance)+", bound2 dis: "+str(bound2Distance) + ", direction: "+str(direction))
    if bound2Distance>bound1Distance and direction == 1:
        return 1 #person enter from boundary 1
    if bound2Distance<bound1Distance and direction == 0:
        return 0 #person enter from boundary 2
def checkLeavingImg(x):
    if x <= 1 or x>=720:
        return 1
    else:
        return 0
# def setId(idName):
#     person_id.append(idName)
# def settempDatabase(uniq_id,direction,boundary):
#     person_info.insert(uniq_id,[direction,boundary])
# def removeId(idName):
#     if idName in person_id:
#         person_id.remove(idName)
def checkDirection(prev_x,cur_x,prev_y,cur_y):
    #print("prev_x"+str(prev_x)+" , cur_x"+str(cur_x))
    deltaX = cur_x-prev_x
    deltaY = cur_y-prev_y
    if deltaX > 0:
        return 1 #walk to right
    if deltaX < 0:
        return 0 #walk to left
while True:
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(int(fps))
    new_frame = time.time()
    fps = int(1/(new_frame-prev_frame))
    prev_frame = new_frame
    #fps calculation
    success, img = cap.read()
    # fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    # cv2.putText(img,str(int(fps)),(960,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    cv2.putText(img, str(fps), (960, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    #detect boundary
    cv2.rectangle(img,(boundary1,0),(boundary2,720),color = (255,255,0),thickness=2)

    #exit boundary
    cv2.line(img,(ex_boundary1,0),(ex_boundary1,720),(255,0,0),thickness=2)
    cv2.line(img,(ex_boundary2,0),(ex_boundary2,720),(255,0,0),thickness=2)

    try:
        classIds, confs, bbox = net.detect(img,confThreshold=threshold)
    except Exception as e:
        print(str(e))
    bbox = list(bbox) #numpy array --> list
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox,confs,threshold,nms_threshold) # non maximum suppression

    for i in indices:
        i = i[0]
        if classIds[i][0] == 1: #person
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            coorXCentroid_float = (x+x+w)/2
            coorYCentroid_float = (y+y+h)/2
            coorXCentroid = int((x+x+w)/2)
            coorYCentroid = int((y+y+h)/2)
            personCentroid = (coorXCentroid,coorYCentroid)
            cv2.circle(img,personCentroid,3,color=(0,0,255),thickness=5)
            cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,"Unique id: {}".format(str(i)).upper(),(box[0]+10,box[1]+60),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
            
            #direction generator (beta mak2) len(tempCoor)>0
            if i in tempCoor.keys():
                #coorXCentroid_prev = tempCoor[i][0]
                #coorYCentroid_prev = tempCoor[i][1]
                coorCentroid_prev = tempCoor.get(i)
                tempCoor.pop(i)
            else:
                coorCentroid_prev = [0,0]
                #coorXCentroid_prev,coorYCentroid_prev=0,0

            direction = checkDirection(coorCentroid_prev[0],coorXCentroid_float,coorCentroid_prev[1],coorYCentroid_float)
            #set previous coordinate
            coorXCentroid_prev = coorXCentroid_float
            coorYCentroid_prev = coorYCentroid_float
            #tempCoor.insert(i,[coorXCentroid_prev,coorYCentroid_prev])
            tempCoor[i] = [coorXCentroid_prev,coorYCentroid_prev]

            #data calculation
            if boundary1<coorXCentroid<boundary2:
                if i in person_dict.keys():
                    continue
                else:
                    enter = checkZoneCrossing(coorXCentroid,coorYCentroid,direction)
                    if enter is None:
                        continue
                    person_dict[i] = enter
            if i in person_dict.keys():
                if coorXCentroid<ex_boundary1 and person_dict.get(i) == 0:
                    person_dict.pop(i)
                    exitCount += 1
                elif coorXCentroid>ex_boundary2 and person_dict.get(i) == 1:
                    person_dict.pop(i)
                    enterCount += 1
                else:
                    continue
            # if (checkLineCrossing(coorXCentroid,offXRef)):
            #     if i in person_id:
            #         continue
            #     else:
            #         enterCount += 1
            #         setId(i)
    #print(person_dict)
    cv2.putText(img, "Entrances: {}".format(str(enterCount)), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
    cv2.putText(img, "Exit: {}".format(str(exitCount)), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
    cv2.imshow("Output",img)
    cv2.waitKey(1)


    # if len(classIds) != 0: #have things to detect
    #     for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    #         if classId == 1: #person
    #             cv2.rectangle(img,box,color=(0,255,0),thickness=2)
    #             cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
    #                         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #             cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
    #                         cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # if boundary1<x<boundary2:
    #             if i in person_id:
    #                 break
    #             else:
    #                 enter = checkZoneCrossing(coorXCentroid,coorYCentroid)
    #                 setId(i)
    #                 settempDatabase(i,direction,enter)

    # if (checkLeavingImg(coorXCentroid) and len(person_id)>0 and len(person_info)>0):
    #             print("people out/in")
    #             if coorXCentroid <= 1 and person_info[i][1]==0:
    #                 print(exitCount)
    #                 exitCount += 1
    #             if coorXCentroid >= 720 and person_info[i][1]==0:
    #                 enterCount += 1
    #             removeId(i)