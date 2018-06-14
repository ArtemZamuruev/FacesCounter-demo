import cv2
import numpy
import config
from pyfacesdk import USize

# Global tracker object
gt_obj = None

def init_settings_window(video_dict):

	global gt_obj

	# Window for setting up a FaceSDK detector
	detset_win_name = "Detector settings"
	cv2.namedWindow(detset_win_name)

	min_dim = min(video_dict["w"], video_dict["h"])

	blank = numpy.zeros((1, 1, 3), numpy.uint8)
	cv2.imshow(detset_win_name, blank)

	cv2.createTrackbar(
		"GT MAX face size",
		detset_win_name,
		gt_obj.maxFaceSize.x, min_dim,
		change_max_face_size)
	cv2.createTrackbar(
		"GT MIN face size",
		detset_win_name,
		gt_obj.minFaceSize.x, min_dim,
		change_min_face_size)

	cv2.createTrackbar(
		"GT MAX lost time",
		detset_win_name,
		gt_obj.maxLostTime, 4000,
		change_gt_max_lost_time)

	cv2.createTrackbar(
		"GT best face scale",
		detset_win_name,
		int(gt_obj.bestFaceScale * 10), 30,
		change_best_face_scale)

	cv2.createTrackbar(
		"GT best frames number",
		detset_win_name,
		gt_obj.bestFramesNum, 5,
		change_best_frames_num)

	cv2.createTrackbar(
		"GT detection filter threshold",
		detset_win_name,
		int(gt_obj.detectionFilterThr * 100), 100,
		change_gt_detection_filter_threshold)

	cv2.createTrackbar(
		"GT detection threshold",
		detset_win_name,
		int(gt_obj.detectionThr * 100), 100,
		change_gt_detection_threshold)

	cv2.createTrackbar(
		"GT match threshold",
		detset_win_name,
		int(gt_obj.trackingThr * 100), 100,
		change_gt_match_threshold)

	cv2.createTrackbar(
		"GT tracking threshold",
		detset_win_name,
		int(gt_obj.trackingThr * 100), 100,
		change_gt_tracking_threshold)





def change_max_face_size(new_val):
	global gt_obj
	gt_obj.maxFaceSize = USize(new_val,new_val)


def change_min_face_size(new_val):
	global gt_obj
	gt_obj.minFaceSize = USize(new_val,new_val)



def change_gt_max_lost_time(new_val):
	global gt_obj
	gt_obj.maxLostTime = new_val


def change_best_face_scale(new_val):
	global gt_obj
	gt_obj.bestFaceScale = float(new_val/10.0)


def change_best_frames_num(new_val):
	global gt_obj
	gt_obj.bestFramesNum = new_val


def change_gt_detection_filter_threshold(new_val):
	global gt_obj
	gt_obj.detectionFilterThr = float(new_val/100.0)


def change_gt_detection_threshold(new_val):
	global gt_obj
	gt_obj.detectionThr = float(new_val/100.0)


def change_gt_match_threshold(new_val):
	global gt_obj
	gt_obj.trackingThr = float(new_val/100.0)


def change_gt_tracking_threshold(new_val):
	global gt_obj
	gt_obj.trackingThr = float(new_val/100.0)
