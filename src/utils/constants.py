# Constants
NETWORK_INPUT_WIDTH = 640
NETWORK_INPUT_HEIGHT = 640
DETECTION_MINIMUM_CONFIDENCE = 0.45
CLASS_SCORE_MINIMUM_CONFIDENCE = 0.5

# Colors
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

# Debug parameters
KEYFRAME_INTERVAL = 4
OBJECT_ID_TO_DEBUG = 0
SLEEP_TIME_IN_SECONDS = 0.1
STOP_FRAME_QUANTITY = 1000  # It is used to stop the video at this frame when SILENT_MODE = True
VIDEO_ACCELERATION_SPEED = 50
DEBUG_MODE = False  # If True, enable more drawings on the video to make debugging easier
SILENT_MODE = False  # If True, run in the console, without graphical interface and stop when it reaches the frame defined by STOP_FRAME_QUANTITY
SHOW_CONFIDENCE = False  # If True, show confidence level next to the object ID.
SLOW_MODE = False  # If True, add a SLEEP_TIME_IN_SECONDS between frames to help with debugging
