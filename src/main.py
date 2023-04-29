import datetime
import os
import time
from collections import defaultdict

import cv2
import dlib
import imutils
import numpy as np
import yolov5
from imutils.video import FPS

from centroid_tracker import CentroidTracker
from count_polylines_intersections import Point, DirectionOptions
from src.utils.constants import *
from src.utils.helper import get_centroid_from_bounding_box, get_rectangle_points_indicating_edge_proximity
from trackable_object import TrackableObject

pt_model_path = os.path.join("net", "model.pt")
# Load custom model
net = yolov5.load(pt_model_path)

# Set model parameters
net.conf = NMS_CONFIDENCE_THRESHOLD  # NMS confidence threshold
net.iou = NMS_IOU_THRESHOLD  # NMS IoU threshold
net.agnostic = False  # NMS class-agnostic
net.multi_label = False  # NMS multiple labels per box
net.max_det = 1000  # Maximum number of detections per image

# Initialize the list of class labels the model was trained to detect
CLASSES = ["person"]


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


def draw_circle_with_id(object_id, bounding_box, centroid, frame):
    text = "ID {}".format(object_id)
    center_x, center_y = centroid
    
    if DEBUG_MODE:
        position = (bounding_box[0] - 10, bounding_box[1] - 10)
    else:
        position = (center_x - 10, center_y - 10)
    
    # Draw both the ID of the object and the centroid of the
    # object on the output frame
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
    cv2.circle(frame, centroid, 4, BLUE, -1)


def convert_polyline_to_array(polyline):
    polyline_points_list = []
    for point in polyline:
        polyline_points_list.append((point.x, point.y))
    return polyline_points_list


def show_info(info, frame):
    frame_width = frame.shape[1]
    # Loop over the info tuples and draw them on our frame
    # enumerate = iterator on (index, value)
    for (i, (key, value)) in enumerate(info):
        text = "{}: {}".format(key, value)
        cv2.putText(frame, text, (frame_width - 135, 60 - (i * 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)


def show_fps(fps_start_time, fps_end_time, total_frames, frame):
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)
    
    fps_text = "FPS: {:.2f}".format(fps)
    
    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, YELLOW, 2)


def get_default_line(width, height):
    point_1 = Point(0.05 * width, 0.2 * height)
    point_2 = Point(0.15 * width, height // 2)
    point_3 = Point(0.85 * width, height // 2)
    point_4 = Point(0.95 * width, 0.2 * height)
    default_line = [point_1, point_2, point_3, point_4]
    return default_line


def main():
    video_path = os.path.join("../videos", VIDEO_FILE)
    cap = cv2.VideoCapture(video_path)
    
    # Initialize the list of object trackers and corresponding class
    # labels
    trackers = []
    labels = []
    
    trackable_objects = {}
    
    # Start the frames per second throughput estimator
    fps_lib = FPS().start()
    
    fps_start_time = datetime.datetime.now()
    total_frames = 0
    frame_height = None
    frame_width = None
    
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
    
    # Initialize the default line and centroid tracker
    default_line = []
    centroid_tracker = CentroidTracker(default_line)
    default_line_points_list = convert_polyline_to_array(default_line)
    
    while True:
        
        if not SILENT_MODE and SLOW_MODE:
            time.sleep(SLEEP_TIME_IN_SECONDS)
        
        # Grab the next frame from the video file
        was_grabbed, frame = cap.read()
        
        # Check to see if we have reached the end of the video file
        if frame is None:
            break
        
        if not SILENT_MODE:
            key = cv2.pollKey()
            # If the `q` key was pressed, break from the loop
            if key == ord('q'):
                break
            handle_video_options(key, cap, is_to_accelerate_video, is_to_move_forward, acceleration_frames_count)
        
        # Resize the frame for faster processing
        frame = imutils.resize(frame, width=320)
        
        # If the frame dimensions are empty, set them
        if frame_height is None or frame_width is None:
            (frame_height, frame_width) = frame.shape[:2]
        
        if total_frames == 0:
            print('frame.shape (frame_height, frame_width)', (frame_height, frame_width))
            default_line = get_default_line(frame_width, frame_height)
            centroid_tracker = CentroidTracker(default_line)
            default_line_points_list = convert_polyline_to_array(default_line)
        
        # Initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []
        
        # Check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if (len(trackers) == 0 and total_frames % (KEYFRAME_INTERVAL / 2) == 0) or (
                len(trackers) != 0 and total_frames % KEYFRAME_INTERVAL == 0):
            # Set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []
            
            # Pass the frame through the network and obtain the detections
            person_detections = net(frame)
            
            # Loop over the detections
            for detection in person_detections.pred:
                # Extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detection[:, 4]
                
                # Filter out frames without detections
                if confidence.nelement() == 0:
                    continue
                
                # Extract the index of the class label from the
                # detections list
                class_label_index = detection[:, 5]
                
                for index in range(confidence.nelement()):
                    label = CLASSES[int(class_label_index[index])]
                    
                    # If the class label is not a person, ignore it
                    if label != "person":
                        continue
                    
                    # Compute the (x, y)-coordinates of the bounding box
                    # for the object
                    person_box = detection[:, :4]
                    
                    start_x, start_y, end_x, end_y = person_box[index]
                    start_x, start_y, end_x, end_y = int(start_x.item()), int(start_y.item()), int(end_x.item()), int(
                        end_y.item())
                    
                    # Convert the frame from BGR to RGB ordering (dlib needs RGB ordering)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation tracker
                    rect = dlib.rectangle(start_x, start_y, end_x, end_y)
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(rgb_frame, rect)
                    
                    # Update our set of trackers and corresponding class
                    # labels
                    labels.append(label)
                    trackers.append(tracker)
                    
                    bounding_box = start_x, start_y, end_x, end_y
                    rects.append(bounding_box)
                    
                    if not SILENT_MODE and DEBUG_MODE and SHOW_CONFIDENCE:
                        cv2.putText(frame, str(round(confidence[index].item(), 2)), (start_x, start_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 2)
        
        # Otherwise, we've already performed detection so let's track multiple objects
        # We should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # Loop over each of the trackers and corresponding labels
            for (tracker, label) in zip(trackers, labels):
                # Set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"
                
                # Convert the frame from BGR to RGB ordering (dlib needs RGB ordering)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update the tracker and grab the position of the tracked object
                tracker.update(rgb_frame)
                tracker_position = tracker.get_position()
                
                # Unpack the position object
                start_x = int(tracker_position.left())
                start_y = int(tracker_position.top())
                end_x = int(tracker_position.right())
                end_y = int(tracker_position.bottom())
                bounding_box = (start_x, start_y, end_x, end_y)
                
                # Add the bounding box coordinates to the rectangles list
                rects.append(bounding_box)
        
        if not SILENT_MODE:
            # Draw a yellow polyline in the center of the frame -- once an
            # object crosses this polyline we will determine whether it was
            # moving 'up' or 'down'
            cv2.polylines(frame, np.int32([default_line_points_list]), False, YELLOW, 2)
            
            # Draw an orange rectangle to indicate that if an object is outside of it, that object is considered to be
            # close to the edges of the video frame. It's used for a visual representation of an improvement made in the
            # centroid tracker.
            rectangle_edge_proximity = get_rectangle_points_indicating_edge_proximity(frame_width, frame_height)
            cv2.rectangle(frame, (rectangle_edge_proximity[0], rectangle_edge_proximity[1]),
                          (rectangle_edge_proximity[2], rectangle_edge_proximity[3]),
                          ORANGE, 2)
        
        bounding_boxes = np.array(rects)
        
        if not SILENT_MODE and DEBUG_MODE:
            for bounding_box in bounding_boxes:
                cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), GREEN, 2)
        
        # Use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects, intersections_list = centroid_tracker.update(bounding_boxes, trackable_objects,
                                                              (frame_width, frame_height))
        
        for (object_id, intersections_count, direction) in intersections_list:
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
            centroid = get_centroid_from_bounding_box(bounding_box)
            center_x, center_y = centroid
            
            centroid_dict[object_id].append(centroid)
            
            if object_id not in object_ids_list:
                object_ids_list.append(object_id)
                start_point = centroid
                end_point = centroid
                cv2.circle(frame, start_point, 4, RED, -1)
                cv2.line(frame, start_point, end_point, RED, 2)
            else:
                length = len(centroid_dict[object_id])
                for point in range(length):
                    if not point + 1 == length:
                        start_point = (centroid_dict[object_id][point][0], centroid_dict[object_id][point][1])
                        end_point = (centroid_dict[object_id][point + 1][0], centroid_dict[object_id][point + 1][1])
                        cv2.circle(frame, centroid, 4, RED, -1)
                        cv2.line(frame, start_point, end_point, RED, 2)
            
            # Check to see if a trackable object exists for the current
            # object ID
            to = trackable_objects.get(object_id, None)
            
            # If there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(object_id, centroid)
            
            # Otherwise, there is a trackable object, so we can add a new centroid to it
            else:
                to.centroids.append(Point(center_x, center_y))
            
            # Store the trackable object in our dictionary
            trackable_objects[object_id] = to
            
            if not SILENT_MODE:
                draw_circle_with_id(object_id, bounding_box, centroid, frame)
        
        # Construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", total_up),
            ("Down", total_down),
            ("Count", total_count),
            ("Status", status),
        ]
        
        fps_end_time = datetime.datetime.now()
        
        if not SILENT_MODE:
            show_info(info, frame)
            show_fps(fps_start_time, fps_end_time, total_frames, frame)
            
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
