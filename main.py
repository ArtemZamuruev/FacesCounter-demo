import time
import json
import sys
import cv2
import os.path as osp
import numpy as np
from pyfacesdk import *

import camtools
import facetools
import utils
import config
import guitools


def main():
	'''
		All the magic starts here.
	'''
	cam = camtools.CamToolkit(config.video_source)
	ftool = facetools.FaceToolkit(cam.video_dict)

	# ======= Initializing GUI =======
	guitools.gt_obj = ftool.g_tracker
	guitools.init_settings_window(cam.video_dict)

	# ============ GUI ===============

	frame = cam.get_frame()

	time_step = int(1000 / cam.fps)
	cur_timestamp = 0

	while not frame is None:
		
		# Get new frame from camera
		frame = cam.get_frame()
		frame8u = Image8U(frame)

		# Updating timestamp for global tracker
		cur_timestamp += time_step

		# Traking faces globally
		ftool.g_tracker.process(frame8u, cur_timestamp)

		tracklets = ftool.g_tracker.tracklets


		tracklets_info = {}

		for i, tlet in enumerate(tracklets):
			
			# print ("Tracklet %d: start time = %s, state = %s" % (i, str(cur_timestamp - int(tlet.start_time)), str(tlet.state)))
			# If tracklet age more than N seconds and it's alive
			# We must do following things:
			# 	0. Check for previous visits
			# 	1. Give it ID
			# 	2. Remember when it happens
			# 	3. Set counter += 1
			# 	4. Calcualte it's descriptor
			#	5. Store it's descriptor and dadta in RAM and disk
			#	6. Store it's best frames
			# 	7. Calculate 


			tlet_age = cur_timestamp - int(tlet.start_time)
			tlet_state = str(tlet.state)

			frame = utils.draw_tracklet_on_frame(
				frame,
				tlet,
				tlet_age=tlet_age,
				tlet_state=tlet_state)


			tlet_key = "Person %d" % tlet.id 
			tracklets_info.update({
				tlet_key :  
					{
						"sep1": "___",
						"Attributes: ": "",
					}
				})

			attributes = ftool.get_attributes(frame, facerect=tlet)
			tracklets_info[tlet_key].update(attributes)

			
			tracklets_info[tlet_key].update({
					"sep2": "___",
					"Demographics: ": "",
				})
			demographics = ftool.get_demographics(frame, facerect=tlet)
			tracklets_info[tlet_key].update(demographics)


			emotion = ftool.get_emotion(frame, facerect=tlet)
			tracklets_info[tlet_key].update({
					"sep3": "___",
					"Emotion": emotion
				})
			
			info_bar = utils.create_blank_image(
				int(frame.shape[0]),
				int(frame.shape[1] * 0.4))
			info_bar = utils.put_text_from_dict_on_image(info_bar, tracklets_info[tlet_key], "left-top")



		frame = np.concatenate((info_bar, frame), axis=1)
		cv2.imshow("Frames from camera", frame)

		pressed_key = cv2.waitKey(time_step)




if __name__ == "__main__":
	main()