import cv2
import config
import numpy as np
from pyfacesdk import *
import utils


class FaceToolkit:
    '''
        Incapsulate tools for face detection
        and recognition

        argument video_params:
            - dict
            - format:
                {
                    "w": width,
                    "h": height,
                    "fps": fps,
                    "roi": (x1, y1, x2, y2),
                }
    '''


    def __init__(self, video_params):
        utils.log("-" * 30)
        utils.log("Start initializing FaceSDK toolkit container:")

        self.video = video_params
        self.vw = self.video["w"]
        self.vh = self.video["h"]
        self.vfps = self.video["fps"]       

        r_x, r_y, r_w, r_h = utils.getRoiOrDefault(self.video) 
        d_min_w ,d_min_h = utils.getMinSizeOrDefault(self.video)
        d_max_w, d_max_h = utils.getMaxSizeOrDefault(self.video)
        
        self.roi = URect(r_x, r_y, r_w, r_h)
        self.minsize = USize(d_min_w, d_min_h)
        self.maxsize = USize(d_max_w, d_max_h)
        self.g_tracker = IGlobalTracker()

        # Set Global tracker parameters:     
        self.g_tracker.maxFaceSize = self.maxsize
        self.g_tracker.minFaceSize = self.minsize
        self.g_tracker.roi = self.roi
        self.g_tracker.maxLostTime = config.tracker_max_lost_time
        self.g_tracker.bestFaceScale = config.tracker_best_face_scale
        self.g_tracker.bestFramesNum = config.tracker_best_frames_num
        self.g_tracker.detectionAlg = DETECTOR_TYPE.ALG1
        self.g_tracker.detectionFilter = config.tracker_detection_filter
        self.g_tracker.detectionFilterThr = config.tracker_detection_filter_threshold
        self.g_tracker.detectionPeriod = config.tracker_detection_period
        self.g_tracker.detectionThr = config.tracker_detection_threshold
        self.g_tracker.favorLongTracks = config.tracker_favor_long_tracks
        self.g_tracker.matchThr = config.tracker_matching_thresholder
        self.g_tracker.trackingThr = config.tracker_matching_threshold
        # -----------------------------

        self.precision_level = config.detector_precision_level
        self.points_detector = IPointsDetector()
        self.extractor = IExtractor()
        self.detector = IDetector()
        self.matcher = IMatcher()

        self.quality_evaluator = IQualityEvaluator()
        self.quality_evaluator.config.blurrines = True;
        self.quality_evaluator.config.angles = True;
        self.quality_evaluator.config.resolution =  True;

        self.attr_classifier = IAttributes()
        self.demo_classifier = IDemographics()
        self.emo_classifier = IEmotions()


    def detect_faces_on_image(self, img):
        if not type(img) is Image8U:
             img = Image8U(img)
        faces = self.detector.detect(
            img,
            self.roi,
            self.minsize,
            self.maxsize,
            self.precision_level)
        return faces


    def extract_face_descriptor(self, img, facerect=None):
        # If facerect is None then we 
        # suppose that we got a cropped image
        # That is why we should initialize a facerect
        # as full frame border
        if not type(img) is Image8U:
            img = Image8U(img)

        if facerect is None:
            x = y = 0
            w = img.width
            h = img.width
            facerect = IRect(x, y, h, w)

        points = self.points_detector.detectFromBbox(img, facerect)
        descriptor = self.extractor.enroll(img, points)
        return descriptor


    def match_desc2desc(self, desc1, desc2):
        return self.matcher.match(desc1, desc2)


    def match_img2desc(self, desc, img, facerect):
        img_desc = self.extract_face_descriptor(img, facerect)
        return self.match_desc2desc(desc, img_desc)

    def match_img2img(selg, img1, facerect1, img2, facerect2):
        img_desc1 = self.extract_face_descriptor(img1, facerect1)
        img_desc2 = self.extract_face_descriptor(img2, facerect2)
        return self.match_desc2desc(img_desc1, img_desc2)


    def recognize_person(self, faceimg, dataset):
        scores = {}
        face_descriptor = self.extract_face_descriptor(faceimg)
        for pers in dataset["persons"]:
            pers_id = pers["id"]
            sum_score = 0.0
        
            for desc in pers["descriptors"]:
                sum_score += self.match_desc2desc(desc, face_descriptor)
            avg_score = sum_score / len(pers["descriptors"])

            if avg_score < 0.65:
                continue

            scores.update({avg_score: pers["name"]})

        if len(scores.keys()) < 1:
            return {"name" : "unknown", "score": 0.0}
        max_score = max(scores.keys())
        max_scored_name = scores[max_score]
        return {"name": max_scored_name, "score": max_score}


    def evaluate_quality(self, img, facerect=None):
        img8u = Image8U(img)

        if facerect is None:
            facerect = utils.img2irect(img)
        elif type(facerect) is Tracklet:
            facerect = utils.tlet2irect(facerect)
        quality = self.quality_evaluator.evaluate(img8u, facerect)




    def get_emotion(self, img, points=None, facerect=None):
        # Parameters: 
        #     img - full frame with detection or cropped with face (not Image8U)
        #     points - antopologic points
        #     facerect - IRect object or Tracklet
        img8u = Image8U(img)

        if points is None:

            if facerect is None:
                facerect = utils.img2irect(img)
            elif type(facerect) is Tracklet:
                facerect = utils.tlet2irect(facerect)
            points = self.points_detector.detectFromBbox(img8u, facerect)

        emotions = self.emo_classifier.classify(img8u, points)
        return emotions.value


    def get_demographics(self, img, points=None, facerect=None):
        img8u = Image8U(img)
        if points is None:
            if facerect is None:
                facerect = utils.img2irect(img)
            elif type(facerect) is Tracklet:
                facerect = utils.tlet2irect(facerect)

            points = self.points_detector.detectFromBbox(img8u, facerect)
        demographics = self.demo_classifier.classify(img8u, points)
        gender = demographics.gender.value.name
        age = int(demographics.age.value)
        ethnicity = demographics.ethnicity.value.name
        demo_dict = {
            "gender": gender,
            "age": age,
            "ethnicity": ethnicity
        }
        return demo_dict


    def get_attributes(self, img, points=None, facerect=None):
        img8u = Image8U(img)
        # if antropologic points are not already calculated:
        if points is None:
            # To caclculate points we should know a rect.
            # If rect is None, then we suppose 
            # that image is already cropped to a face size
            # and rectangle of interest is a full image shape
            if facerect is None:
                facerect = utils.img2irect(img)
            elif type(facerect) is Tracklet:
                facerect = utils.tlet2irect(facerect)

            points = self.points_detector.detectFromBbox(img8u, facerect)

        attributes = self.attr_classifier.classify(img8u, points)

        facial_hair = attributes.facial_hair.value.name
        glasses = attributes.glasses.value.name
        hair_color = attributes.hair_color.value.name
        hair_type = attributes.hair_type.value.name
        headwear = attributes.headwear.value.name

        attr_dict = {
            "facial_hair": facial_hair,
            "hair_color": hair_color,
            "hair_type": hair_type,
            "headwear": headwear,
            "glasses": glasses
        }
        return attr_dict