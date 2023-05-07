# Constants
VIDEO_FILE = "ceiling_camera.mp4"
MINIMUM_DETECTION_CONFIDENCE = 0.65
NMS_IOU_THRESHOLD = 0.45
CENTROID_TRACKER_MAXIMUM_DISAPPEARED = 30
CENTROID_TRACKER_MAXIMUM_DISTANCE = 70
KEYFRAME_INTERVAL = 4

# Colors
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 140, 255)

# Debug parameters
SLEEP_TIME_IN_SECONDS = 0.1  # It is used to slow down the video when SLOW_MODE = True
STOP_FRAME_QUANTITY = 1000  # It is used to stop the video at this frame when SILENT_MODE = True
VIDEO_ACCELERATION_SPEED = 25  # It is used to control the video speed when moving forward or backward
DEBUG_MODE = True  # If True, enable more drawings on the video to make debugging easier
SILENT_MODE = False  # If True, run in the console, without graphical interface and stop when it reaches the frame defined by STOP_FRAME_QUANTITY
SHOW_CONFIDENCE = False  # If True, print the object detection confidence to the console
SLOW_MODE = False  # If True, add a SLEEP_TIME_IN_SECONDS between frames to help with debugging
