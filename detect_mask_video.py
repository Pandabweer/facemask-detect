import matplotlib.pyplot as plt
import numpy as np
import asyncio
import imutils
import typing
import keras
import time
import cv2
import os

from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.engine.functional import Functional

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from imutils.video.webcamvideostream import WebcamVideoStream
from imutils.video import VideoStream
from imutils import paths

# set error logging to exception only, define the paths
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
prototxt_path = './face_detector/deploy.prototxt'
weights_path = './face_detector/res10_300x300_ssd_iter_140000.caffemodel'

# initialize the initial learning rate, number of epochs to train for, and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = os.getcwd() + "/dataset"
CATEGORIES = ["with_mask", "without_mask"]

class Train_mask_detector(object):
	def __init__(self, *, graph: bool = False) -> None:
		self.graph = graph
		self.aug = self.data_aug()
		self.model = self.cons_model()
		self.compile_model()
		self.H = self.training_network()
		self.finalize()

	def load_images(self) -> tuple:
		print("[INFO] loading images...")
		data, labels = [], []

		for category in CATEGORIES:
		    path = os.path.join(DIRECTORY, category)
		    for img in os.listdir(path):
		    	img_path = os.path.join(path, img)
		    	image = load_img(img_path, target_size=(224, 224))
		    	image = img_to_array(image)
		    	image = preprocess_input(image)

		    	data.append(image)
		    	labels.append(category)

		return data, labels

	def lable_encoding(self) -> tuple:
		data, labels = self.load_images()
		data = np.array(data, dtype="float32")
		labels = np.array(to_categorical(
			LabelBinarizer().fit_transform(labels)
		))

		return train_test_split(
			data, labels, test_size=0.20, stratify=labels, random_state=42
		)

	def data_aug(self) -> ImageDataGenerator:
		return ImageDataGenerator(
			rotation_range=20,
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
			fill_mode="nearest"
		)

	def cons_model(self) -> Model:
		baseModel = MobileNetV2(
			weights="imagenet",
			include_top=False,
			input_tensor=Input(shape=(224, 224, 3))
		)

		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(128, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation="softmax")(headModel)

		model = Model(inputs=baseModel.input, outputs=headModel)

		for layer in baseModel.layers:
			layer.trainable = False

		return model

	def compile_model(self) -> None:
		print("[INFO] compiling model...")
		opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
		self.model.compile(
			loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
		)

	def training_network(self) -> tuple:
		trainX, testX, trainY, testY = self.lable_encoding()
		return self.model.fit(
			self.aug.flow(trainX, trainY, batch_size=BS),
			steps_per_epoch=len(trainX) // BS,
			validation_data=(testX, testY),
			validation_steps=len(testX) // BS,
			epochs=EPOCHS
		)

	def finalize(self) -> None:
		# make predictions on the testing set
		print("[INFO] evaluating network...")
		predIdxs = model.predict(testX, batch_size=BS)

		# for each image in the testing set we need to find the index of the
		# label with corresponding largest predicted probability
		predIdxs = np.argmax(predIdxs, axis=1)

		# show a nicely formatted classification report
		print(classification_report(
				testY.argmax(axis=1), predIdxs, target_names=lb.classes_)
		)

		print("[INFO] saving mask detector model...")
		model.save("./face_detector/mask_detector.model", save_format="h5")

		if self.graph:
			self.make_graph()

	def make_graph():
		# plot the training loss and accuracy
		N = EPOCHS
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig("plot.png")

class Main(object):
	def __init__(self, prototxt_path: str, weights_path: str) -> None:
		self.face_net = cv2.dnn.readNet(prototxt_path, weights_path)
		self.mask_net = load_model('./face_detector/mask_detector.model')
		self.vs = asyncio.run(self.start_video())
		asyncio.run(self.detect_mask())

	async def start_video(self) -> WebcamVideoStream:
		print('[INFO] Video starten...')
		return VideoStream(src=0).start()

	async def detect_and_predict_mask(
		self,
		frame: np.ndarray,
		face_net: cv2.dnn_Net,
		mask_net: Functional
	) -> typing.Union[tuple, None]:
		h, w = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(
			frame, 1.0, (224, 224), (104.0, 177.0, 123.0)
		)

		face_net.setInput(blob)
		detections = face_net.forward()
		faces, locs, preds = [], [], []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence
			confidence = detections[0, 0, i, 2]

			if confidence > 0.5:
				# compute the (x, y)-coordinates
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				startX, startY, endX, endY = box.astype('int')

				# ordering, resize it to 224x224, and preprocess it
				face = preprocess_input(img_to_array(cv2.resize(cv2.cvtColor(
					frame[
						max(0, startY):min(h - 1, endY),
						max(0, startX):min(w - 1, endX)
					],
					cv2.COLOR_BGR2RGB), (224, 224)))
				)

				# add the face and bounding boxes to their respective lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			faces = np.array(faces, dtype='float32')
			preds = mask_net.predict(faces, batch_size=32)

		# return a 2-tuple of the face locations and their locations
		return locs, preds

	async def detect_mask(self) -> None:
		print('[INFO] Programma is gestart!')
		while True:
			# video of 1920 pixels
			frame = imutils.resize(self.vs.read(), width=1920)

			# detect faces and if they are wearing a face mask or not
			locs, preds = await self.detect_and_predict_mask(
				frame, self.face_net, self.mask_net
			)

			# loop over the detected face locations and their locations
			for box, pred in zip(locs, preds):
				# unpack the box and predictions
				startX, startY, endX, endY = box
				mask, withoutMask = pred

				label = ('Masker' if mask > withoutMask else 'Geen masker')
				color = (0, 255, 0) if label == 'Masker' else (0, 0, 255)

				# include the probability in the label
				label = '{}: {:.2f}%'.format(label, max(mask, withoutMask) * 100)

				cv2.putText(
					frame, label, (startX, startY - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2
				)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

			# show the output frame
			cv2.imshow('Frame', frame)

			# tick (fps) -> 60/(0.04*60) = 25 fps
			#time.sleep(0.03)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				print('[INFO] Programma beëindigd')

				cv2.destroyAllWindows()
				self.vs.stop()
				break

if __name__ == '__main__':
	print('[INFO] Programma is aan het opstarten...')
	if not os.path.isfile('./face_detector/mask_detector.model'):
		print('[INFO] No data module found, constructing module, this may take awhile')
		Train_mask_detector(graph=False)

	Main(prototxt_path, weights_path)
