import numpy as np
from datetime import datetime as dt
import config
import facetools as ft
from pyfacesdk import *
import json
import os.path as osp
import cv2


def log(message, offset=0, log_in_file=False):
	event_time = dt.now().time().isoformat()[0:-4]
	print("%s%s%s" % (event_time, " " + "\t" * offset, message))
	
	if log_in_file:
		log_lile = open(config.log_file, 'a')
		log_file.writeline("%s%s%s" % (event_time, " " + "\t" * offset, message))
		log_file.close()


def random_color(id):
    np.random.seed(id)
    return tuple(np.random.randint(0, 255, 3).tolist())



# ==========================================================
# FUNCTIONS TO WORK WITH CONFIGS

def getMinSizeOrDefault(video_params):

    if config.det_minsize is None:
        log("Absolute det_minsize value not found in config. Using det_minsize_scale", offset=1)
        if config.det_minsize_scale is None:
            log("det_minsize_scale parameter was not found in config file", offset=1)
            det_minsize_scale = 0.05
            log("Set default to %.2f" % det_minsize_scale, offset=1)
        else:
            det_minsize_scale = config.det_minsize_scale
        d_min_w = video_params["w"] * det_minsize_scale
        d_min_h = video_params["h"] * det_minsize_scale
    else:
        d_min_w, d_min_h = config.det_minsize
    log("Detection min size is %d x %d" % (int(d_min_w), int(d_min_h)), offset=1)
    log("\n", offset=1)
    return (int(d_min_w), int(d_min_h))


def getMaxSizeOrDefault(video_params):

    if config.det_maxsize is None:
        log("Absolute det_maxsize value not found in config. Using det_maxsize_scale", offset=1)
        if config.det_maxsize_scale is None:
            log("det_maxsize_scale parameter was not found in config file", offset=1)
            det_maxsize_scale = 0.95
            log("Set default to %.2f" % det_maxsize_scale, offset=1)
        else:
            det_maxsize_scale = config.det_maxsize_scale
        d_max_w = video_params["w"] * det_maxsize_scale
        d_max_h = video_params["h"] * det_maxsize_scale

    else:
        d_max_w, d_max_h = config.det_maxsize
    log("Detection max size is %d x %d" % (int(d_max_w), int(d_max_h)), offset=1)
    log("\n", offset=1)
    return (int(d_max_w), int(d_max_h))


def getRoiOrDefault(video_params):
    if video_params["roi"] is None:
        log("ROI is set to full frame", offset=1)
        x = y = 0
        w = video_params["w"]
        h = video_params["h"]
    else:
        log("ROI was set to %s\n" % str(video_params["roi"]), offset=1)
        x, y, w, h = video_params["roi"]

    return (x, y, w, h)


# ===========================================================


def read_persons_data():
    persons_file = open(config.persons_json_path, 'r')
    persons_data = json.load(persons_file)
    persons_file.close()
    for pers in persons_data["persons"]:
        
        descriptors_filename = osp.join(
            config.descriptors_dir,
            "%s.descriptor" % pers["id"])

        descriptors_file = open(descriptors_filename, 'r')
        descriptors_data = json.load(descriptors_file)
        descriptors_file.close()

        pers.update({"descriptor_file": descriptors_filename})
        pers.update(
            {"descriptors": 
                [list2desc(desc_list) for desc_list in descriptors_data["descriptors"]]
            })

    log("Persons data read OK")

    return persons_data


def draw_tracklet_on_frame(
    frame,
    tlet,
    color=None,
    tlet_age=None,
    tlet_state=None,
    tlet_name=None):


    if color is None:
        color = random_color(tlet.id)
    face = tlet.position
    x, y = face.x, face.y
    w, h = face.w, face.h

    pos_ltop = (x, y)
    pos_rbot = (w + x, h + y)

    tness= 4
    cv2.rectangle(
        frame,
        pos_ltop,
        pos_rbot,
        color,
        tness)

    if not tlet_state is None:
        cv2.putText(
            frame,
            str(tlet_state),
            (x, y + h + 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            color,
            2)


    if not tlet_state is None:
        cv2.putText(
            frame,
            str(tlet_age),
            (x, y + h + 45),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            color,
            2)

    if not tlet_name is None:
        cv2.putText(
            frame,
            str(tlet_name),
            (x, y - 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            color,
            2)

    return frame


def put_text_from_dict_on_image(img, lines_dict, alignment):

    w, h = img.shape[0], img.shape[1]
    padding = config.info_panel_padding


    if alignment == "left-top":
        x_0, y_0 = padding, padding
        y_step = config.font_size + config.space_between_text_line

    lines_counter = 0
    
    for key, val in lines_dict.items():
        if val == "___":
            line = "_" * 10
        else:
            line = "%s: %s" % (key, str(val))

        y_i  = y_0 + y_step * lines_counter
        
        img = cv2.putText(
            img,
            line,
            (x_0, y_i),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            config.font_color,
            config.font_thickness)

        lines_counter += 1
    return img

def list2desc(d_list):
    '''
    
    Converts list of base64 numbers
    to FaceSDK Descriptor object

    '''
    d_barr = bytearray(
        np.array(
            d_list, np.uint8).tobytes())
    return Descriptor(d_barr)


def tlet2irect(tlet):
    '''
        Creating a IRect object from a tracklet object
    '''
    x, y = tlet.position.x, tlet.position.y
    h, w = tlet.position.h, tlet.position.w
    return IRect(x, y, w, h)


def img2irect(img):
    '''
        Creating and returns a IRect object which is 
        a full shape of the image.

        Use it when you are working with cropped parts of image
    '''
    if type(img) is np.ndarray:
        w, h = img.shape[0], img.shape[1]
    elif type(img) is Image8U:
        w, h = img.widht, img.height
    return IRect(0, 0, w, h)


def create_blank_image(h, w, elements=0):
    if elements:
        return np.ones((h, w, 3), np.uint8)
    else:
        return np.zeros((h, w, 3), np.uint8)

