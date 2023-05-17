import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
import torch

VIDEO_PATH = "data\cauthang.mp4"
# Define Poligon Postion
area = np.array([[1, 140], [117, 211], [125, 291], [
                49, 713], [46, 794], [1, 792]], np.int32)

# area = np.array([[0, 152], [106, 217], [111, 284], [
#                 37, 707], [35, 781], [1, 782],[2, 759], [17, 759], 
#                 [18, 709], [95, 281],[88, 224], [3, 168]], 
#                 np.int32)

ALARM_BUFFER = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# region Initialize YOLO model
model = YOLO('model\yolov8n.pt', device)

label = 'person'
label_index = model.names.get(label)
# endregion Initialize YOLO model

# region Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5,
                    model_complexity=1)

# endregion Initialize Mediapipe Pose model

prev_frame_time = 0
new_frame_time = 0

count_alarm_buffer = 0

cap = cv2.VideoCapture(VIDEO_PATH)

# Create windown
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    # region Frame read
    ret, frame = cap.read()

    if not ret:
        if cap.get(cv2.CAP_PROP_BUFFERSIZE) == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ????
            print("Camera Not ret")
        else:
            print("Video Not ret, set frame of video to first")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    # endregion Frame read

    # region Detect object
    results_obj = model.predict(source=frame,
                                classes=0,
                                conf=0.3,
                                max_det=3,
                                save=False, save_txt=False)
    
    # endregion Detect object

    # region Detect pose
    humans_boxes = results_obj[0].boxes
    for human_box in humans_boxes:
        box = human_box.xyxy.tolist()[0]
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        human = frame[int(y1):int(y2), int(x1):int(x2)]

        frame_rgb = cv2.cvtColor(human, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_draw = mp.solutions.drawing_utils
            mp_draw.draw_landmarks(
                human, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
          
            frame[int(y1):int(y2), int(x1):int(x2)] = human

            # region Alarm
            hand_landmarks = results.pose_landmarks.landmark[15:23]

            # Count number of hand landmarks within the polygon area
            count_point_area = 0
            for point in hand_landmarks:
                pt = (int((point.x*human.shape[1]) + x1),
                      int((point.y*human.shape[0]) + y1))
                cv2.circle(frame, pt, 5, (255, 255, 0), thickness=-1)
                if cv2.pointPolygonTest(area, pt, False) >= 0:
                    count_point_area += 1

            # Update alarm buffer
            if count_point_area <= 0:
                count_alarm_buffer += 1
            else:
                count_alarm_buffer = 0

            # Draw safety status on the frame
            if count_alarm_buffer >= ALARM_BUFFER:
                cv2.putText(frame, "NGUY HIEM", (width-200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "AN TOAN", (width-200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            if cv2.waitKey(1) & 0xFF == ord("s"):    
                mp_draw.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            # endregion Alarm
    # endregion Detect pose

    # region Draw
    # Draw polylines
    cv2.polylines(frame, [area], True, (0, 255, 255), 2)
    # cv2.fillPoly(frame, [area], (0, 255, 255))
    frame = results_obj[0].plot()

    # Write FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, "FPS: {:.2f}".format(
        fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("YOLOv8 Inference", frame)
    # endregion Draw

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
