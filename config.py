import os.path as osp
import os


app_name = "Face recognizer"


# Debug Mode switcher
# True or False
# If True - additional info output
debug_mode = True


# Path to log file
log_file = "./log_ex.log"

write_videolog = False
videolog_filename = "out_videolog.avi"


# ======================
# Location settings    |
# ======================

base_dir = os.getcwd()
persons_json_path = osp.join(base_dir, "storage/persons.json")
descriptors_dir = osp.join(base_dir, "storage/descriptors/")
photos_dir = osp.join(base_dir, "storage/photos/")
bestframes_dir = osp.join(base_dir, "storage/bestframes/")


# Filenames patterns: 
#   - photo: <id><n>.jpg
#   - descriptor <id>.descr




# =======================
# Video settings        |
# =======================
video_scale = 1.0

# Video Source 
# 0 - use directly connected camera (USB web-cam etc.)
# "rtsp://<username>:<password>@<address>:<port>/<url>" - for IP-cameras (RTSP stream)
# Also you can put a path of video file stored locally

# video_source = "rtsp://admin:vlabsadmin1235@192.168.253.233:554/h264_2"
video_source = 0
# video_source = "./test_videos/600.part4.mkv"


# =======================
# FaceSDK settings      |
# =======================

global_tracker_max_lost_time = 1000
# detector_confidence = 0.2
# detection_filter_confidence = 0.5

det_minsize = None
det_minsize_scale = 0.1

det_maxsize = None
det_maxsize_scale = 0.8


tracker_max_lost_time = 2500
tracker_best_face_scale = 2.0
tracker_best_frames_num = 4

tracker_detection_filter = True
tracker_detection_filter_threshold = 0.4


tracker_detection_period = 10
tracker_favor_long_tracks = True

tracker_detection_threshold = 0.4

tracker_matching_thresholder = True
tracker_matching_threshold = 0.4

detector_precision_level = 0.6



# =======================
# Draw settings         |
# =======================

info_panel_padding = 20
space_between_text_line = 8
font_size = 12
font_thickness = 1
font_color = (220, 220, 220)