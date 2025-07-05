
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from playsound import playsound

CLASSES = ["aeroplane", "bicycle", "boat",
	"bottle", "bus", "car", "cat", "diningtable", "motorbike", 
	"sofa", "train", "tvmonitor","Plastic cover","bag","Mobile phone","Bracelet","Glass bottle","Nails",
        "Metal items","Pen","Marker","Duster","Chocolate wrappers","Medicine","Chair",
        "System","Mouse","Keyboard","Id card","Watches","Belt","Shoes","Clothes","Rock"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.5:
			idx = int(detections[0, 0, i, 1])
			if idx in [3,8,11,15,16,9]:
                                continue
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			playsound('beep.mpeg')

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
