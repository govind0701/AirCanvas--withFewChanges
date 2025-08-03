
# All the imports go here
import cv2
from datetime import datetime
import numpy as np
import mediapipe as mp
from collections import deque


# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
sbpoints = [deque(maxlen=1024)]
gdpoints= [deque(maxlen=1024)]
bkpoints= [deque(maxlen=1024)]
erase_mode = False


# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
sb_index=0
bk_index=0
gd_index=0
framecount=0 #for undoredo optimization
#The kernel to be used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = [
     (255, 0, 0),       # Blue
    (0, 255, 0),        # Green
     (0, 0, 255),       # Red
    (0, 255, 255),      # Yellow
    (135, 206, 235),    # Sky Blue
    (255, 215, 0),      # Gold
    (0, 0, 0),          # Black
]
colorIndex = 0
bg_colors = [(255, 255, 255),(255, 255, 255),(255, 255, 255), (255, 0, 0), (255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255),(0,255,255),(0,255,255),(0,255,255)]
bg_color_index = 0  # Default to white

# Here is code for Canvas setup
paintWindow = np.full((471, 636, 3), bg_colors[bg_color_index], dtype=np.uint8)
paintWindow = cv2.rectangle(paintWindow, (20,1), (100,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (120,1), (200,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (220,1), (300,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (320,1), (400,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (420,1), (500,65), (0,255,255), 2)
paintWindow = cv2.rectangle(paintWindow, (520,1), (600,65), colors[colorIndex], 2)
paintWindow = cv2.rectangle(paintWindow, (20,80), (100,130), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (20,145), (57,195), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (63,145), (100,195), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (20, 210), (100, 260), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (20,275), (100,325), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (20,340), (100,390), (0, 0, 0), 2)


cv2.putText(paintWindow, "CLEAR", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (130, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (230, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (330, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (430, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "OTHER", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[colorIndex], 2, cv2.LINE_AA)
cv2.putText(paintWindow, "Re", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "Un", (68, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BG_Col", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "Erase", (30, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "Save", (30,  305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "Quit", (30, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# History stacks for Undo/Redo functionality
undo_stack = []
redo_stack = []
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape
    framecount=framecount+1
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (20,1), (100,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (120,1), (200,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (220,1), (300,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (320,1), (400,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (420,1), (500,65), (0,255,255), 2)
    frame = cv2.rectangle(frame, (520,1), (600,65), colors[colorIndex], 2)
    frame = cv2.rectangle(frame, (20,80), (100,130), (0,0,0), 2)
    frame = cv2.rectangle(frame,(20,145), (57,195), (0,0,0), 2)
    frame = cv2.rectangle(frame,(63,145), (100,195), (0,0,0), 2)
    frame = cv2.rectangle(frame,(20,210), (100,260), (0,0,0), 2)
    frame = cv2.rectangle(frame,(20,275), (100,325), (0,0,0), 2)
    frame = cv2.rectangle(frame, (20,340), (100,390), (0,0,0), 2)
    
    cv2.putText(frame, "CLEAR", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (130, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (230, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (330, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (430, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "OTHER", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[colorIndex], 2, cv2.LINE_AA)
    cv2.putText(frame, "BG_Col", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Re", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Un", (68, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Erase", (30,235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Save", (30, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Quit", (30, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) 
    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        #middle_finger=(landmarks[12][0],landmarks[12][1])
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, fore_finger, 3, (0,255,0),-1)
        #print(fore_finger[1]-thumb[1]) #removed to exesess prints in console
        
        if (thumb[1]-fore_finger[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            sbpoints.append(deque(maxlen=512))
            sb_index += 1
            gdpoints.append(deque(maxlen=512))
            gd_index += 1
            bkpoints.append(deque(maxlen=512))
            bk_index += 1

        elif fore_finger[1] <= 68:
            erase_mode=False
            if 20 <= fore_finger[0] <= 100: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                sbpoints= [deque(maxlen=512)]
                bkpoints= [deque(maxlen=512)]
                gdpoints= [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                sb_index=0
                bk_index=0
                gd_index=0
                

                paintWindow[67:,102:,:] = 255
            elif 120 <= fore_finger[0] <= 200:
                    #erase_mode=False
                    colorIndex = 0 # Blue
            elif 220 <= fore_finger[0] <= 300:
                    #erase_mode=False
                    colorIndex = 1 # Green
            elif 320 <= fore_finger[0] <= 400:
                    #erase_mode=False
                    colorIndex = 2 # Red
            elif 420 <= fore_finger[0] <= 500:
                    #erase_mode=False
                    colorIndex = 3 # Yellow
            elif 520<= fore_finger[0]<=600:
                    #erase_mode=False
                    colorIndex= (colorIndex+1)% len(colors)                    
        elif  fore_finger[0] <= 102:
            erase_mode= False
            if  80<fore_finger[1]<130:#BackgroundColor
                erase_mode= False
                bg_color_index = (bg_color_index + 1) % len(bg_colors)  # Cycle through background colors
                paintWindow[67:,102:] = bg_colors[bg_color_index] 
            elif 210 <= fore_finger[1] <= 260 : #erase
                erase_mode = True
            elif 275<= fore_finger[1] <= 325 : #save
                erase_mode= False
                filename = f"drawing{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, paintWindow)
            elif 340<= fore_finger[1]<=390: #quit
                erase_mode= False
                filename = f"drawing{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, paintWindow)
                ret=False   
            elif 145 <= fore_finger[1] <= 195:  # Undo Redo Button
                erase_mode = False
                # When undoing
                if 63<=fore_finger[0]<=100:
                    if undo_stack:
                    # Save current state to redo stack
                     redo_stack.append((paintWindow.copy(), [bpoints.copy(), gpoints.copy(), rpoints.copy(), ypoints.copy(),sbpoints.copy(),gdpoints.copy(),bkpoints.copy()],[blue_index,green_index,red_index,yellow_index,sb_index,gd_index,bk_index]))
                    # Restore previous state
                     paintWindow, (bpoints, gpoints, rpoints, ypoints,sbpoints,gdpoints,bkpoints),(blue_index,green_index,red_index,yellow_index,sb_index,gd_index,bk_index) = undo_stack.pop()
                elif 10 <= fore_finger[0] <= 58:  # Redo Button
                #erase_mode = False
                # When redoing
                  if redo_stack :
                    # Save current state to undo stack before restoring redo state
                   undo_stack.append((paintWindow.copy(), [bpoints.copy(), gpoints.copy(), rpoints.copy(), ypoints.copy(),sbpoints.copy(),gdpoints.copy(),bkpoints.copy()],[blue_index,green_index,red_index,yellow_index,sb_index,gd_index,bk_index]))
                    # Restore redo state
                   paintWindow, (bpoints, gpoints, rpoints, ypoints,sbpoints,gdpoints,bkpoints),(blue_index,green_index,red_index,yellow_index,sb_index,gd_index,bk_index) = redo_stack.pop()
        else :
            if erase_mode:
                cv2.circle(paintWindow, fore_finger, 20, (255, 255, 255), -1)
                for points_list in [bpoints, gpoints, rpoints, ypoints,sbpoints,gdpoints,bkpoints]:
                    for point_deque in points_list:
                        for point in list(point_deque):  # Convert deque to list to allow safe iteration
                            if (fore_finger[0] - point[0])**2 + (fore_finger[1] - point[1])**2 <= 400:
                                point_deque.remove(point)
                paintWindow[67:,102:] = bg_colors[bg_color_index]
                paintWindow = cv2.rectangle(paintWindow, (20,1), (100,65), (0,0,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (120,1), (200,65), (255,0,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (220,1), (300,65), (0,255,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (320,1), (400,65), (0,0,255), 2)
                paintWindow = cv2.rectangle(paintWindow, (420,1), (500,65), (0,255,255), 2)
                paintWindow = cv2.rectangle(paintWindow, (520,1), (600,65), colors[colorIndex], 2)
                paintWindow = cv2.rectangle(paintWindow, (20,80), (100,130), (0,0,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (20,145), (57,195), (0,0,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (63,145), (100,195), (0,0,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (20, 210), (100, 260), (0, 0, 0), 2)
                paintWindow = cv2.rectangle(paintWindow, (20,275), (100,325), (0,0,0), 2)
                paintWindow = cv2.rectangle(paintWindow, (20,340), (100,390), (0, 0, 0), 2)


                cv2.putText(paintWindow, "CLEAR", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "BLUE", (130, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "GREEN", (230, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "RED", (330, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "YELLOW", (430, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "OTHER", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[colorIndex], 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "Re", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "Un", (68, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "BG_Col", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "Erase", (30, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "Save", (30,  305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(paintWindow, "Quit", (30, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE) 
                
                '''elif colorIndex in [0, 1, 2, 3]:  # If drawing is happening
                if not erase_mode:  # Store state only if new stroke starts
                    undo_stack.append((paintWindow.copy(), [bpoints.copy(), gpoints.copy(), rpoints.copy(), ypoints.copy(),sbpoints.copy(),gdpoints.copy(),bkpoints.copy()]))
                    redo_stack.clear()  # Clear redo when new drawing happens
                    if len(undo_stack) > 50:  # Limit memory usage
                        undo_stack.pop(0)'''
            elif colorIndex == 0:
                bpoints[blue_index].appendleft(fore_finger)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(fore_finger)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(fore_finger)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(fore_finger)
            elif colorIndex == 4:
                sbpoints[sb_index].appendleft(fore_finger)
            elif colorIndex == 5:
                gdpoints[gd_index].appendleft(fore_finger)
            elif colorIndex == 6:
                bkpoints[bk_index].appendleft(fore_finger)
                
    # Append the next deques when nothing is detected to avoid messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
        sbpoints.append(deque(maxlen=512))
        sb_index += 1
        gdpoints.append(deque(maxlen=512))
        gd_index += 1
        bkpoints.append(deque(maxlen=512))
        bk_index += 1
        
    
    
    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints,sbpoints,gdpoints,bkpoints]
    # for j in range(len(points[0])):
    #         for k in range(1, len(points[0][j])):
    #             if points[0][j][k - 1] is None or points[0][j][k] is None:
    #                 continue
    #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
    

    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)
    key=cv2.waitKey(1)
    if key == ord('q'):
        filename = f"drawing{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, paintWindow)
        break
    elif key== ord('s'):
        filename = f"drawing{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, paintWindow)
    # Inside the main loop, when handling drawing
    elif colorIndex in [0, 1, 2, 3,4,5,6]:  # If drawing is happening
                if  framecount%10==0 and not erase_mode :  # Store state only if new stroke starts
                    undo_stack.append((paintWindow.copy(), [bpoints.copy(), gpoints.copy(), rpoints.copy(), ypoints.copy(),sbpoints.copy(),gdpoints.copy(),bkpoints.copy()],[blue_index,green_index,red_index,yellow_index,sb_index,gd_index,bk_index]))
                    #redo_stack.clear()  # Clear redo when new drawing happens
                    if len(undo_stack) > 60:  # Limit memory usage
                        undo_stack.pop(0)
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
