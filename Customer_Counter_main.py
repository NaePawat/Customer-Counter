import cv2
import numpy as np
import time
from PiVideoStream import PiVideoStream
import imutils

class Detector:
    def __init__ (self):
        self.totalCount = 0
        
        self.threshold = 0.45
        self.nms_threshold = 0.5
        self.enterCount = 0
        self.exitCount = 0
        self.id_count = 0
        self.offsetRefLines = 150
        self.offXRef = 600

        self.classNames = []
        self.person_dict = dict()
        self.tempCoor = dict()
        
        self.width = 608
        self.height = 400

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,self.width)
        self.cap.set(4,self.height)
        self.cap.set(10,60)
        self.boundary1 = int(0.2*self.width)
        self.boundary2 = int(0.8*self.width)
        self.ex_boundary1 = int(0.2*self.width)
        self.ex_boundary2 = int(0.8*self.width)
        classFile = 'coco.names'
        with open(classFile,'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'
        self.net = cv2.dnn_DetectionModel(weightsPath,configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/ 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
    def resetTotalCount(self):
        self.totalCount = 0
    def getTotalCount(self):
        return self.totalCount
    def checkLineCrossing(self,x,coorXLine):
        absDistance = abs(x-coorXLine)
        if absDistance<=2:
            return 1
        else:
            return 0
    def checkZoneCrossing(self,x,y,direction):
        bound1Distance = abs(x-self.boundary1)
        bound2Distance = abs(x-self.boundary2)
        #print("bound1 dis: "+str(bound1Distance)+", bound2 dis: "+str(bound2Distance) + ", direction: "+str(direction))
        if bound2Distance>bound1Distance and direction == 1:
            return 1 #person enter from boundary 1
        if bound2Distance<bound1Distance and direction == 0:
            return 0 #person enter from boundary 2
    def checkLeavingImg(self,x):
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
    def checkDirection(self,prev_x,cur_x,prev_y,cur_y):
        #print("prev_x"+str(prev_x)+" , cur_x"+str(cur_x))
        deltaX = cur_x-prev_x
        deltaY = cur_y-prev_y
        if deltaX > 0:
            return 1 #walk to right
        if deltaX < 0:
            return 0 #walk to left
    def run(self):
        prev_frame = 0
        new_frame = 0
        start_time = 0
        delay_time = 2
        vs = PiVideoStream().start()
        time.sleep(2.0)
        while True:
            #fps calculation
            img = vs.read()
            img = imutils.resize(img, width=self.width)
            
            new_frame = time.time()
            fps = int(1/(new_frame-prev_frame))
            prev_frame = new_frame
            cv2.putText(img ,str(fps),(0,80),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
            
            if start_time == 0:
                start_time = time.time()
            elapsed_time = time.time() - start_time
            if elapsed_time >= delay_time:
                print("hello")
                start_time = 0

            #detect boundary
            #cv2.line(img,(offXRef,0),(offXRef,720),(255,0,0),thickness=2)
            cv2.line(img,(self.boundary1,0),(self.boundary1,self.height),(255,255,0),thickness=2)
            cv2.line(img,(self.boundary2,0),(self.boundary2,self.height),(255,255,0),thickness=2)

            #exit boundary
            cv2.line(img,(self.ex_boundary1,0),(self.ex_boundary1,self.height),(255,0,0),thickness=2)
            cv2.line(img,(self.ex_boundary2,0),(self.ex_boundary2,self.height),(255,0,0),thickness=2)

            try:
                classIds, confs, bbox = self.net.detect(img,confThreshold=self.threshold)
            except Exception as e:
                print(str(e))
            bbox = list(bbox) #numpy array --> list
            confs = list(np.array(confs).reshape(1,-1)[0])
            confs = list(map(float,confs))

            indices = cv2.dnn.NMSBoxes(bbox,confs,self.threshold,self.nms_threshold) # non maximum suppression

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
                    cv2.circle(img,personCentroid,3,color=(0,0,255),thickness=3)
                    cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=1)
                    cv2.putText(img,self.classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,"Unique id: {}".format(str(i)).upper(),(box[0]+10,box[1]+60),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 
                    
                    #direction generator (beta mak2) len(tempCoor)>0
                    if i in self.tempCoor.keys():
                        #coorXCentroid_prev = tempCoor[i][0]
                        #coorYCentroid_prev = tempCoor[i][1]
                        coorCentroid_prev = self.tempCoor.get(i)
                        self.tempCoor.pop(i)
                    else:
                        coorCentroid_prev = [0,0]
                        #coorXCentroid_prev,coorYCentroid_prev=0,0

                    direction = self.checkDirection(coorCentroid_prev[0],coorXCentroid_float,coorCentroid_prev[1],coorYCentroid_float)
                    #set previous coordinate
                    coorXCentroid_prev = coorXCentroid_float
                    coorYCentroid_prev = coorYCentroid_float
                    #tempCoor.insert(i,[coorXCentroid_prev,coorYCentroid_prev])
                    self.tempCoor[i] = [coorXCentroid_prev,coorYCentroid_prev]

                    #data calculation
                    if self.boundary1<coorXCentroid<self.boundary2:
                        if i in self.person_dict.keys():
                            continue
                        else:
                            enter = self.checkZoneCrossing(coorXCentroid,coorYCentroid,direction)
                            if enter is None:
                                continue
                            self.person_dict[i] = enter
                    if i in self.person_dict.keys():
                        if coorXCentroid<self.ex_boundary1 and self.person_dict.get(i) == 0:
                            self.person_dict.pop(i)
                            self.exitCount += 1
                        elif coorXCentroid>self.ex_boundary2 and self.person_dict.get(i) == 1:
                            self.person_dict.pop(i)
                            self.enterCount += 1
                        else:
                            continue
                    # if (checkLineCrossing(coorXCentroid,offXRef)):
                    #     if i in person_id:
                    #         continue
                    #     else:
                    #         enterCount += 1
                    #         setId(i)
#             print(self.person_dict)
            
            sumCount = self.enterCount - self.exitCount
            self.totalCount += sumCount
            
            cv2.putText(img, "Entrances: {}".format(str(self.enterCount)), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
            cv2.putText(img, "Exit: {}".format(str(self.exitCount)), (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 0, 1), 2)
            cv2.imshow("Output",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        vs.stop()
        
detect = Detector()
detect.run()