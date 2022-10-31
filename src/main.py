import datetime
import os
import time
from collections import defaultdict

import cv2
import dlib
import numpy as np
from imutils.video import FPS

from centroid_tracker import CentroidTracker
from count_polylines_intersections import Point, DirectionOptions
from src.utils.constants import *
from trackable_object import TrackableObject

protopath = os.path.join("net", "MobileNetSSD_deploy.prototxt")
modelpath = os.path.join("net", "MobileNetSSD_deploy.caffemodel")

# Load the serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

# Only enable it if you are using OpenVino environment
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

pt1 = Point(10, 40)
pt2 = Point(40, 120)
pt3 = Point(280, 120)
pt4 = Point(310, 40)
default_line = [pt1, pt2, pt3, pt4]

centroid_tracker = CentroidTracker(default_line)


def non_max_suppression_fast(boxes, overlap_threshold):
    try:
        if len(boxes) == 0:
            return []
        
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        
        pick = []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlap_threshold)[0])))
        
        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression: {}".format(e))


def handle_video_options(key, cap, is_to_accelerate_video, is_to_move_forward, acceleration_frames_count):
    # If the `s` key was pressed, stop the video acceleration
    if key == ord('s'):
        is_to_accelerate_video[0] = False
    
    # If the `f` key was pressed, the video moves forward
    if key == ord('f'):
        is_to_accelerate_video[0] = True
        is_to_move_forward[0] = True
    
    # If the `b` key was pressed, the video moves backward
    if key == ord('b'):
        is_to_accelerate_video[0] = True
        is_to_move_forward[0] = False
    
    if is_to_accelerate_video[0]:
        if is_to_move_forward[0]:
            acceleration_frames_count[0] += VIDEO_ACCELERATION_SPEED
        elif acceleration_frames_count[0] > 0:
            acceleration_frames_count[0] -= VIDEO_ACCELERATION_SPEED
        
        cap.set(1, acceleration_frames_count[0])


def draw_circle_with_id(to, object_id, bounding_box, c_x, c_y, frame):
    if not to.pre_counted and DEBUG_MODE:
        # Draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "NOT COUNTED ID {}".format(object_id)
        cv2.putText(frame, text, (bounding_box[0] - 10, bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
        cv2.circle(frame, (c_x, c_y), 4, BLUE, -1)
    
    if not to.pre_counted and not DEBUG_MODE:
        text = "ID {}".format(object_id)
        cv2.putText(frame, text, (c_x - 10, c_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
        cv2.circle(frame, (c_x, c_y), 4, BLUE, -1)
        if object_id == OBJECT_ID_TO_DEBUG:
            print(text, '| Counted =', to.pre_counted, ' | Centroids (c_x, c_y):', (c_x, c_y))
    
    if to.pre_counted and DEBUG_MODE:
        # Draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "COUNTED ID {}".format(object_id)
        cv2.putText(frame, text, (bounding_box[0] - 10, bounding_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
        cv2.circle(frame, (c_x, c_y), 4, YELLOW, -1)
    
    if to.pre_counted and not DEBUG_MODE:
        text = "ID {}".format(object_id)
        cv2.putText(frame, text, (c_x - 10, c_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
        cv2.circle(frame, (c_x, c_y), 4, YELLOW, -1)
        if object_id == OBJECT_ID_TO_DEBUG:
            print(text, '| Counted =', to.pre_counted, ' | Centroids (c_x, c_y):', (c_x, c_y))


def convert_polyline_to_array(polyline):
    polyline_points_list = []
    for point in polyline:
        polyline_points_list.append((point.x, point.y))
    return polyline_points_list


def show_info(info, frame):
    frame_height = frame.shape[0]
    # Loop over the info tuples and draw them on our frame
    # enumerate = iterator on (index, value)
    for (i, (key, value)) in enumerate(info):
        text = "{}: {}".format(key, value)
        cv2.putText(frame, text, (10, frame_height - ((i * 20) + 20) - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)


def show_fps(fps_start_time, fps_end_time, total_frames, frame):
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)
    
    fps_text = "FPS: {:.2f}".format(fps)
    
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, GREEN, 2)


def show_people_count(count, count_type, position, frame):
    count_txt = count_type + ": {}".format(count)
    cv2.putText(frame, count_txt, position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BLUE, 2)


def main():
    cap = cv2.VideoCapture('../videos/ceiling_camera.mp4')
    
    # Initialize the list of object trackers and corresponding class
    # labels
    trackers = []
    labels = []
    
    trackable_objects = {}
    
    # Start the frames per second throughput estimator
    fps_lib = FPS().start()
    
    fps_start_time = datetime.datetime.now()
    total_frames = 0
    
    total_down = 0
    total_up = 0
    total_count = 0
    
    total_up_ids_list = []
    total_down_ids_list = []
    total_count_ids_list = []
    
    acceleration_frames_count = [0]
    is_to_move_forward = [False]
    is_to_accelerate_video = [False]
    
    centroid_dict = defaultdict(list)
    object_ids_list = []
    
    default_line_points_list = convert_polyline_to_array(default_line)
    
    while True:
        
        if not SILENT_MODE and SLOW_MODE:
            time.sleep(SLEEP_TIME_IN_SECONDS)
        
        # Grab the next frame from the video file
        was_grabbed, frame = cap.read()
        
        if not SILENT_MODE:
            key = cv2.pollKey()
            # If the `q` key was pressed, break from the loop
            if key == ord('q'):
                break
            handle_video_options(key, cap, is_to_accelerate_video, is_to_move_forward, acceleration_frames_count)
        
        # Check to see if we have reached the end of the video file
        if frame is None:
            break
        
        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=320)
        
        # Grab the frame dimensions
        (H, W) = frame.shape[:2]
        
        # Convert the frame from BGR to RGB ordering (dlib needs RGB ordering)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []
        
        # If our correlation object tracker list is empty we first need to
        # apply an object detector to seed the tracker with something
        # to actually track
        # Check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if len(trackers) == 0 or total_frames % KEYFRAME_INTERVAL == 0:
            
            # Set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []
            
            # Convert the frame to a blob
            blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (INPUT_WIDTH, INPUT_HEIGHT), (0, 0, 0), swapRB=True,
                                         crop=False)
            
            # Pass the blob through the network and obtain the detections
            # and predictions
            net.setInput(blob)
            person_detections = net.forward()
            
            # Loop over the detections
            for i in np.arange(0, person_detections.shape[2]):
                # Extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = person_detections[0, 0, i, 2]
                
                # Filter out weak detections by requiring a minimum
                # confidence
                if confidence > MINIMUM_CONFIDENCE:
                    # Extract the index of the class label from the
                    # detections list
                    class_label_index = int(person_detections[0, 0, i, 1])
                    
                    label = CLASSES[class_label_index]
                    
                    # If the class label is not a person, ignore it
                    if label != "person":
                        continue
                    
                    # Compute the (x, y)-coordinates of the bounding box
                    # for the object
                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (start_x, start_y, end_x, end_y) = person_box.astype("int")
                    
                    # Construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                    tracker.start_track(rgb, rect)
                    
                    # Update our set of trackers and corresponding class
                    # labels
                    labels.append(label)
                    trackers.append(tracker)
                    
                    rects.append((start_x, start_y, end_x, end_y))
                    
                    if not SILENT_MODE and DEBUG_MODE:
                        # Draw the bounding box and text for the object
                        cv2.rectangle(frame, (start_x, start_y), (start_x + end_x, start_y + end_y), GREEN, 2)
                        if SHOW_CONFIDENCE:
                            cv2.putText(frame, str(confidence), (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        GREEN, 2)
        
        # Otherwise, we've already performed detection so let's track multiple objects
        # We should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # Loop over each of the trackers and corresponding labels
            for (tracker, label) in zip(trackers, labels):
                # Set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"
                
                # Update the tracker and grab the position of the tracked object
                tracker.update(rgb)
                tracker_position = tracker.get_position()
                
                # Unpack the position object
                start_x = int(tracker_position.left())
                start_y = int(tracker_position.top())
                end_x = int(tracker_position.right())
                end_y = int(tracker_position.bottom())
                
                # Add the bounding box coordinates to the rectangles list
                rects.append((start_x, start_y, end_x, end_y))
                
                if not SILENT_MODE and DEBUG_MODE:
                    # Draw the bounding box from the correlation object tracker
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                                  RED, 2)
        
        if not SILENT_MODE:
            # Draw a yellow polyline in the center of the frame -- once an
            # object crosses this polyline we will determine whether it was
            # moving 'up' or 'down'
            cv2.polylines(frame, np.int32([default_line_points_list]), False, YELLOW, 2)
        
        bounding_boxes = np.array(rects)
        bounding_boxes = bounding_boxes.astype(int)
        rects = non_max_suppression_fast(bounding_boxes, 0.3)
        
        # Use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects, intersections_list = centroid_tracker.update(rects, trackable_objects)
        
        for (object_id, intersections_count, direction) in intersections_list:
            print('(object_id, intersections_count, direction)', (object_id, intersections_count, direction))
            if direction is None:
                continue
            if intersections_count % 2 != 0:
                total_count += 1
                total_count_ids_list.append(object_id)
                print('Counted: ', object_id)
                
                if direction == DirectionOptions.UP:
                    print('UP | object_id: ', object_id)
                    total_up += 1
                    total_up_ids_list.append(object_id)
                else:
                    print('DOWN | object_id: ', object_id)
                    total_down += 1
                    total_down_ids_list.append(object_id)
        
        # Loop over the tracked objects
        # objects.items() = iterable views on associations
        for (object_id, bounding_box) in objects.items():
            x1, y1, x2, y2 = bounding_box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            
            c_x = int((x1 + x2) / 2.0)
            c_y = int((y1 + y2) / 2.0)
            
            centroid_dict[object_id].append((c_x, c_y))
            
            if object_id not in object_ids_list:
                object_ids_list.append(object_id)
                start_point = (c_x, c_y)
                end_point = (c_x, c_y)
                cv2.circle(frame, start_point, 4, RED, -1)
                cv2.line(frame, start_point, end_point, RED, 2)
            else:
                length = len(centroid_dict[object_id])
                for point in range(length):
                    if not point + 1 == length:
                        start_point = (centroid_dict[object_id][point][0], centroid_dict[object_id][point][1])
                        end_point = (centroid_dict[object_id][point + 1][0], centroid_dict[object_id][point + 1][1])
                        cv2.circle(frame, (c_x, c_y), 4, RED, -1)
                        cv2.line(frame, start_point, end_point, RED, 2)
            
            # Check to see if a trackable object exists for the current
            # object ID
            to = trackable_objects.get(object_id, None)
            
            # If there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(object_id, (c_x, c_y))
            
            # Otherwise, there is a trackable object, so we can utilize it
            # to determine direction
            else:
                # The difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c.y for c in to.centroids]
                pre_direction = c_y - np.mean(y)
                to.centroids.append(Point(c_x, c_y))
                
                # Check whether the object has been pre-counted or not.
                # Pre-counting is useful for the centroid tracker to avoid setting the same object ID if two objects
                # appear next to each other at the same time. Therefore, if the object was already pre-counted, its id
                # cannot be reassigned and the centroid tracker recognizes the trackable object as another object with
                # a new ID.
                if not to.pre_counted:
                    # If the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if pre_direction < 0 and c_y < H // 2:
                        to.pre_counted = True
                    
                    # If the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif pre_direction > 0 and c_y > H // 2:
                        to.pre_counted = True
            
            # Store the trackable object in our dictionary
            trackable_objects[object_id] = to
            
            if not SILENT_MODE:
                draw_circle_with_id(to, object_id, bounding_box, c_x, c_y, frame)
        
        # Construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", total_up),
            ("Down", total_down),
            ("Count", total_count),
            ("Status", status),
        ]
        
        fps_end_time = datetime.datetime.now()
        lpc_count = len(objects)
        opc_count = len(trackable_objects)
        
        if not SILENT_MODE:
            show_info(info, frame)
            show_fps(fps_start_time, fps_end_time, total_frames, frame)
            show_people_count(lpc_count, "LPC", (5, 60), frame)
            show_people_count(opc_count, "OPC", (5, 90), frame)
            
            # Show the output frame
            cv2.imshow("Application", frame)
        else:
            if total_frames == STOP_FRAME_QUANTITY:
                break
        
        total_frames += 1
        
        # Update the FPS counter
        fps_lib.update()
    
    # Stop the timer and display FPS information
    fps_lib.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps_lib.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps_lib.fps()))
    print("[INFO] approx. total_frames: ", total_frames)
    print("[INFO] approx. KEYFRAME_INTERVAL: ", KEYFRAME_INTERVAL)
    print("[INFO] Final list of IDs counted up (total_up_ids_list): ", total_up_ids_list)
    print("[INFO] Final list of IDs counted down (total_down_ids_list): ", total_down_ids_list)
    print("[INFO] Final list of IDs counted (total_count_ids_list): ", total_count_ids_list)
    
    cv2.destroyAllWindows()
    cap.release()


main()
