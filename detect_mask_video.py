import numpy as np
import asyncio
import imutils
import typing
import keras
import time
import cv2
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.engine.functional import Functional
from imutils.video import VideoStream

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
prototxt_path = './face_detector/deploy.prototxt'
weights_path = './face_detector/res10_300x300_ssd_iter_140000.caffemodel'

class Main(object):
	def __init__(self, prototxt_path: str, weights_path: str) -> None:
		self.face_net = cv2.dnn.readNet(prototxt_path, weights_path)
		self.mask_net = load_model('./face_detector/mask_detector.model')
		self.vs = asyncio.run(self.start_video())
		asyncio.run(self.detect_mask())

	async def start_video(self) -> imutils.video.webcamvideostream.WebcamVideoStream:
		print('[INFO] Video starten...')
		return VideoStream(src=0).start()

	async def detect_and_predict_mask(self, frame: np.ndarray, face_net: cv2.dnn_Net, mask_net: Functional) -> typing.Union[tuple, None]:
		# grab the dimensions of the frame and then construct a blob from it
		h, w = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the face detections
		face_net.setInput(blob)
		detections = face_net.forward()

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locs = []
		preds = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				startX, startY, endX, endY = box.astype('int')

				# ensure the bounding boxes fall within the dimensions of the frame
				startX, startY = (max(0, startX), max(0, startY))
				endX, endY = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			faces = np.array(faces, dtype='float32')
			preds = mask_net.predict(faces, batch_size=32)

		# return a 2-tuple of the face locations and their corresponding locations
		return (locs, preds)

	async def detect_mask(self) -> None:
		print('[INFO] Programma is gestart!')
		# loop over the frames from the video stream
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 1920 pixels
			frame = self.vs.read()
			frame = imutils.resize(frame, width=1920)

			# detect faces in the frame and determine if they are wearing a face mask or not
			(locs, preds) = await self.detect_and_predict_mask(frame, self.face_net, self.mask_net)

			# loop over the detected face locations and their corresponding locations
			for (box, pred) in zip(locs, preds):
				# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred

				# determine the class label and color we'll use to draw the bounding box and text
				label = 'Masker' if mask > withoutMask else 'Geen masker'
				color = (0, 255, 0) if label == 'Masker' else (0, 0, 255)

				# include the probability in the label
				label = '{}: {:.2f}%'.format(label, max(mask, withoutMask) * 100)

				# display the label and bounding box rectangle on the output frame
				cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			# show the output frame
			cv2.imshow('Frame', frame)

			# tick (fps) -> 60/(0.04*60) = 25 fps
			time.sleep(0.03)

			# if the `q` key was pressed, break from the loop
			if cv2.waitKey(1) & 0xFF == ord('q'):
				print('[INFO] Programma beëindigd')

				# do a bit of cleanup
				cv2.destroyAllWindows()
				self.vs.stop()
				break

if __name__ == '__main__':
	print('[INFO] Programma is aan het opstarten...')
	Main(prototxt_path, weights_path)
