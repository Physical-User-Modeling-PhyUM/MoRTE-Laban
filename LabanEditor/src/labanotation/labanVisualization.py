# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------------------------

import cv2
import numpy as np

class convertLabanScriptToView:
    cnt = 0
    width = 480
    height = 960
    bottom = 160
    scale = 100 # pixels/second
    img = np.array((height, width))
    name = ''
    timeOffset = 0.0
    timeScale = 1.0
    duration = 0

    #------------------------------------------------------------------------------
    # class initialization
    # w: width; h: height; text: labanotation script
    #
    def __init__(self, w, h, text):
        if text == "":
            self.timeOffset = 0.0
            self.timeScale = 1.0
            self.width = w
            self.height = h
            self.img = np.ones((self.height, self.width), np.uint8) * 255
            return

        # **Store Labanotation for all body parts**
        
      # Init storage
        l_elbow, l_wrist, r_wrist, r_elbow = [], [], [], []
        r_knee, l_knee, r_foot, l_foot = [], [], [], []
        r_ankle, l_ankle = [], []
        head, r_body, l_body = [], [], []
        support, rotation = [], []
        i = 0
        self.duration = 0
        self.cnt = 0
        self.name = "Labanotation View"

        laban_lines = text.split("\n")


        i = 0
        while i < len(laban_lines):
            line = laban_lines[i].strip().lower()

            if line == "" or line.startswith("#"):
                i += 1
                continue

            tmp = line.split(":")
            if len(tmp) < 2:
                i += 1
                continue

            key = tmp[0]
            time_stamp = None
            if key == "start time":
                if self.cnt == 0:
                    start = int(tmp[1])
                self.cnt += 1
                self.duration = int(tmp[1]) - start
                time_stamp = int(tmp[1]) / 1000.0
            elif key == "duration":
                pass  # already handled via "start time"
            else:
                if time_stamp is None:
                    time_stamp = self.cnt / 1000.0  # fallback if "start time" missing

                # Safely extract direction and level
                direction = tmp[1] if len(tmp) > 1 else ""
                level = tmp[2] if len(tmp) > 2 else ""

                entry = [time_stamp, direction, level]

                # Staff-mapped joint parsing
                if key == "left elbow":
                    l_elbow.append(entry)
                elif key == "left wrist":
                    l_wrist.append(entry)
                elif key == "right wrist":
                    r_wrist.append(entry)
                elif key == "right elbow":
                    r_elbow.append(entry)
                elif key == "left knee":
                    l_knee.append(entry)
                elif key == "right knee":
                    r_knee.append(entry)
                elif key == "left foot":
                    l_foot.append(entry)
                elif key == "right foot":
                    r_foot.append(entry)
                elif key == "left ankle":
                    l_ankle.append(entry)
                elif key == "right ankle":
                    r_ankle.append(entry)
                    
                elif key == "left body":
                    l_body.append(entry)
                elif key == "right body":
                    r_body.append(entry)
                    
                elif key == "head":
                    head.append(entry)
                elif key == "support":
                    support.append([time_stamp, direction])  # direction holds the support label
                elif key == "rotation":
                    rotation.append([time_stamp, level])  # level holds the rotation amount

            i += 1


        self.scale = h // (2 + self.duration / 1000)
        self.width = w
        self.height = h + self.bottom
        self.timeOffset = h
        self.timeScale = self.scale * (self.duration / 1000.0)

        self.img = np.ones((self.height, self.width), np.uint8) * 255
        self.init_canvas()

        # 🟦 Left Side (Staff Left Columns)
        self.draw_limb(1, "left", l_elbow)
        self.draw_limb(2, "left", l_wrist)

        self.draw_limb(3, "left", l_body)  # Could be torso top (upper body)
        self.draw_limb(4, "left", l_ankle)
        self.draw_limb(5, "left", l_foot)
        self.draw_limb(6, "left", l_knee)

        # 🟥 Right Side (Staff Right Columns)
        self.draw_limb(7, "right", r_knee)
        self.draw_limb(8, "right", r_foot)
        self.draw_limb(9, "right", r_ankle)
        self.draw_limb(10, "right", r_body)  # Optional repeat of torso lower
        self.draw_limb(11, "right", r_wrist)
        self.draw_limb(12, "right", r_elbow)
        self.draw_limb(13, "right", head)


     

    #------------------------------------------------------------------------------
    # draw a vertical dashed line.
    def dashed(self, x1, y1, y2):
        dash = 40
        if y1 > y2:
            a = y1; y1 = y2; y2 = a
        for i in range(0,int(int(np.abs(y2-y1))/dash)):
            cv2.line(self.img,(x1,y2-i*dash),(x1,y2-i*dash-int(dash/2)),0,2)
        if y2-(i+1)*dash > y1:
            cv2.line(self.img,(x1,y2-(i+1)*dash),(x1,y1),0,2)
    

    #------------------------------------------------------------------------------
    # canvas initialization
    def init_canvas(self):
        self.unit = int(self.width / 13)  # Updated for 13 columns
        floor = int(self.height - self.bottom)

        # Central 3 staff lines (columns 5–7)
        cv2.line(self.img, (self.unit * 5, 0), (self.unit * 5, floor), 0, 2)
        cv2.line(self.img, (self.unit * 6, 0), (self.unit * 6, floor), 0, 2)
        cv2.line(self.img, (self.unit * 7, 0), (self.unit * 7, floor), 0, 2)

        # Horizontal floor lines
        cv2.line(self.img, (self.unit * 5, floor), (self.unit * 7, floor), 0, 2)
        cv2.line(self.img, (self.unit * 5, floor + 4), (self.unit * 7, floor + 4), 0, 2)

        # Dashed vertical grid lines for all 13 columns
        for i in range(1, 13):
            self.dashed(self.unit * i, 0, floor)

        # Horizontal time ticks (centered at middle column)
        i = 0
        while True:
            x1 = int(self.unit * 6 - 3)
            x2 = int(self.unit * 6 + 2)
            y = int(floor - i * self.scale)
            if y < 0:
                break
            cv2.line(self.img, (x1, y), (x2, y), 0, 2)
            i += 1

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        subtitle = self.height - 50
        title = self.height - 20

        labels = [
            'ElbowL', 'WristL', 'BodyL', 'AnkleL', 'FootL', 'SupportL',
            'SupportR', 'FootR', 'AnkleR', 'BodyR', 'WristR', 'ElbowR', ' Head'
        ]

        alignments = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # text x-offsets

        for i in range(13):
            x = self.unit * i + alignments[i]
            cv2.putText(self.img, labels[i], (x, title), font, 0.5, 1, 2)

        # Name label
        cv2.putText(self.img, self.name, (self.unit * 3 + 30, self.height - 40), font, 1.2, 1, 1)

    #------------------------------------------------------------------------------
    # draw sign of Labanotation.
    # side: right hand side, left hand side
    #     for determin which forward/backward sign should be used.
    # direction: place, 
    #     forward, backward
    #     right, left
    #     right forward (diagonal), right backward (diagonal)
    #     left forward (diagonal), left backward (diagonal)
    # level: low, normal, high
    # (x1,y1) is the left top corner, (x2,y2) is the right bottom corner.
    # 
    def sign(self, cell, timeTuple, side="right", dire = "place", lv = "low"):
        
        
        (time1,time2) = timeTuple
        unit = self.width/13
        x1 = int((cell-1)*unit+5)#left top corner
        x2 = int(cell*unit-5)
        y1 = int(self.height-self.bottom-int(time2*self.scale*15)+5)#right bottom corner
        y2 = int(self.height-self.bottom-int(time1*self.scale*15)-5)
        support= (' o' in lv)
        lv=lv.replace(' o', '')
        #shading: pattern/black/dot
        if lv=="normal":
            cv2.circle(self.img,(int((x1+x2)/2),int((y1+y2)/2)), 4, 0,-1)
        elif lv=="high":
            step = 20
            i=0
            while True:
                xl = int(x1)#start point at the left
                yl = int(y1+i*step)           
                xr = int(x1+i*step) # end point at th right
                yr = int(y1)
                if yl > y2:
                    xl = yl-y2+xl
                    yl = y2
                if xr > x2:
                    yr = y1+xr-x2
                    xr = x2
                if (xl>xr)or(yr>yl):
                    break
                cv2.line(self.img, (xl,yl),(xr, yr),0,2)
                i+=1
        elif lv=="low":
            #cv2.rectangle(self.img,((x1,y1),(x2,y2),0,-1))
            cv2.rectangle(self.img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),-1)
        else:
            print ("Unknown Level: " + lv)
        
        # shape: trapezoid, polygon, triangle, rectangle
        if dire=="right":
            pts = np.array([[x1,y1-1],[x2+1,y1-1],[x2+1,int((y1+y2)/2)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y2+1],[x2+1,y2+1],[x2+1,int((y1+y2)/2)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x1,y2],[x2,int((y1+y2)/2)]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left":
            pts = np.array([[x1-1,y1-1],[x2,y1-1],[x1-1,int((y1+y2)/2)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1-1,y2+1],[x2,y2+1],[x1-1,int((y1+y2)/2)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,(y1+y2)/2],[x2,y1],[x2,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left forward":
            pts = np.array([[x1,y1-1],[x2+1,y1-1],[x2+1,y1+int((y2-y1)/3)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1+int((y2-y1)/3)],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="right forward":
            pts = np.array([[x1-1,y1-1],[x2+1,y1-1],[x1-1,y1+int((y2-y1)/3)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1+int((y2-y1)/3)],[x2,y1],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="left backward":
            pts = np.array([[x1,y2+1],[x2+1,y2+1],[x2+1,y2-int((y2-y1)/3)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2-int((y2-y1)/3)],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="right backward":
            pts = np.array([[x1-1,y2+1],[x2+1,y2+1],[x1-1,y2-int((y2-y1)/3)]],np.int32)
            cv2.fillPoly(self.img,[pts],255)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2-(y2-y1)/3]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="forward" and side=="right":
            cv2.rectangle(self.img,(int(x1+(x2-x1)/2),y1-1),(x2+1,int(y1+(y2-y1)/3)),(255,0,0),-1)
            pts = np.array([[x1,y1],[x1+(x2-x1)/2,y1],[x1+(x2-x1)/2,y1+(y2-y1)/3],
                            [x2,y1+(y2-y1)/3],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="forward" and side=="left":
            cv2.rectangle(self.img,(x1-1,y1-1),(int(x1+(x2-x1)/2),y1+int((y2-y1)/3)),(255,0,0),-1)
            pts = np.array([[x1,y1+(y2-y1)/3],[int(x1+(x2-x1)/2),y1+int((y2-y1)/3)],[x1+(x2-x1)/2,y1],
                            [x2,y1],[x2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="backward" and side=="right":
            cv2.rectangle(self.img,(int(x1+(x2-x1)/2),y2-int((y2-y1)/3)),(x2+1,y2+1),(255,0,0),-1)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2-int((y2-y1)/3)],
                            [int(x1+(x2-x1)/2),y2-int((y2-y1)/3)],[x1+(x2-x1)/2,y2],[x1,y2]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="backward" and side=="left":
            cv2.rectangle(self.img,(x1-1,y2-int((y2-y1)/3)),(int(x1+(x2-x1)/2),y2+1),(255,0,0),-1)
            pts = np.array([[x1,y1],[x2,y1],[x2,y2],
                            [int(x1+(x2-x1)/2),y2],[int(x1+(x2-x1)/2),y2-int((y2-y1)/3)],
                            [x1,y2-(y2-y1)/3]],np.int32)
            cv2.polylines(self.img, [pts], True, 0, 2)
        elif dire=="place":#"Place"
            #print("Error in statement")
            cv2.rectangle(self.img,(x1,y1),(x2,y2),(0,0,0),2)
        else:
            print ("Unknown Direction: " + side + ": " + dire)

        if support:
            center_x = int((x1 + x2) / 2)
            center_y = int(y1 - 2)  # small offset above the symbol
            cv2.circle(self.img, (center_x, center_y), 5, (0, 0, 0), thickness=2)

            # Then draw a smaller white filled circle on top
            cv2.circle(self.img, (center_x, center_y), 4, (255, 255, 255), thickness=-1)
    #------------------------------------------------------------------------------
    # draw one column of labanotation for one limb
    # 
    def draw_limb(self,cell,side,laban):
        self.sign(cell,(-90.0/(15*self.scale),-5.0/(15*self.scale)),side,laban[0][1],laban[0][2])
        i=1
        k=0
        while i <= self.cnt-1:
            if laban[i-1][1]==laban[i][1] and laban[i-1][2]==laban[i][2]:
                pass
            else:
                
                #sign(cell,(time1,time2), side="Right", dire = "Place",lv = "Low"):
                self.sign(cell,(laban[k][0],laban[i][0]),side,laban[i][1],laban[i][2])
                k=i
            i+=1

