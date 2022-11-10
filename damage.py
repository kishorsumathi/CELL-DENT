import cv2
import numpy as np
class Damage_Detection():
            def __init__(self):
                self.net = cv2.dnn.readNet("yolov4-sam-mish_best.weights", "yolov4-sam-mish .cfg")
                self.classes = []
                with open("classes.names", "r") as f:
                    self.classes = [line.strip() for line in f.readlines()]
            def image(self,path):
                        img=cv2.imread(path)
                        model = cv2.dnn_DetectionModel(self.net)
                        model.setInputParams(size=(512,512), scale=1/255, swapRB=True)
                        classes, scores, bbox=model.detect(img, 0.4,0.3)
                        label=[]
                        print(bbox)
                        if len(bbox) !=0:
                                for (label,confidence,bbox) in zip(classes,scores,bbox):
                                    label=self.classes[label]
                                    cv2.rectangle(img,bbox,(255,5,5),thickness=2)
                                    cv2.putText(img, "{} ({:.2f})".format(label,float(confidence)),
                                                    (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                    color=(124,252,0), thickness=2)
                        cv2.imwrite("out.jpg",img)
                        return img,label
            def video(self,path):
                 cap = cv2.VideoCapture(path)
                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                 fps = int(cap.get(cv2.CAP_PROP_FPS))
                 fourcc = cv2.VideoWriter_fourcc(*'XVID')
                 output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
                 if (cap.isOpened()== False):
                        print("Error opening video file") 
                 while cap.isOpened():
                        ret, frame = cap.read()
                        if ret == True:
                            model = cv2.dnn_DetectionModel(self.net)
                            model.setInputParams(size=(512,512), scale=1/255, swapRB=True)
                            classes, scores, bbox=model.detect(frame, 0.3,0.3)
                            print(bbox,classes)
                            print("process going on")
                            label=[]
                            if len(bbox) !=0:
                                    for (label,confidence,bbox) in zip(classes,scores,bbox):
                                        label=self.classes[label]
                                        # if label=="4 mm Dent":
                                        #     cv2.rectangle(frame,bbox,color=(124,252,0),thickness=2)
                                        #     cv2.putText(frame, "Safe ({:.2f})".format(float(confidence)),
                                        #                 (bbox[0],bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        #                 color=(124,252,0), thickness=2)
                                        if label=="2 mm Dent":
                                            cv2.rectangle(frame,bbox,color=(0,0,255),thickness=2)
                                            cv2.putText(frame, "{} ({:.2f})".format(label,float(confidence)),
                                                        (bbox[0],bbox[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                        color=(0,0,255), thickness=2)
                                        elif label=="4 mm Dent":
                                            cv2.rectangle(frame, bbox,color = (255, 0, 0),thickness=2)
                                            cv2.putText(frame, "{} ({:.2f})".format(label,float(confidence)),
                                                        (bbox[0],bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                        color = (255, 0, 0), thickness=2)
                            output.write(frame)
                         

                    
                        else:
                            break
                 cap.release()
                 return "done"