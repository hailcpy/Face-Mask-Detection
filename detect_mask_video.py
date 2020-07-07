# USAGE
# python detect_mask_video.py

# import the necessary packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import pygame
import argparse
import imutils
import time
import cv2
import os

def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = ensure_color(face)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces)>0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=16)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

#defines for display

def mask_sign(mask_detected,tick,cross):
	if mask_detected:
		return tick
	else:
		return cross

def temp_sign(temperature_normal,tick,cross):
	if temperature_normal:
		return tick
	else:
		return cross

def show_mask_text(mask_detected):
	if mask_detected:
		return 'You are wearing a mask'
	else:
		return 'Please wear a mask'
    
def show_temp_text(temperature_normal,temp,normal_temp):
	if temperature_normal:
		return 'Body temperature is Normal:'+str(temp)+ u'\N{DEGREE SIGN}C'
	else:
		return 'Body temperature is High! : '+str(temp)+ u'\N{DEGREE SIGN}C' + '. Normal is '+ str(normal_temp)+ u'\N{DEGREE SIGN}C'
  

def display(mask_detected,temperature_normal,temp):
	all_ok = mask_detected and temperature_normal
	normal_temp = 37.2

	# /////////////////////////////////

	# define locations ////////////
	# bounding box
	width = 1920
	height = 1080

	bbx = int(width/2) - 650
	bby = int(height/2)-100

	bbwidth = 1300
	bbheight = 330

	checkx = 760
	checky = 100

	tempx = bbx+90
	tempy = bby+30

	maskx = bbx+90
	masky = bby+190

	mask_signx = maskx+200
	mask_signy = masky+30

	temp_signx = tempx+200
	temp_signy = tempy+40

	mtextx = mask_signx+100
	mtexty = mask_signy+10

	ttextx = temp_signx+100
	ttexty = temp_signy+10

	# //////////////////

	# initialize//////////////////////////////
	# create the screen
	screen = pygame.display.set_mode((width,height))
	# initialize font
	font = pygame.font.Font('./saved/font/LEMONMILK-Regular.otf',32)

	# ///////////////////////////////////////


	# load images title and font /////////////////////////////////////
	#  TODO add condition if image exists to prevent crashing
	temperature = pygame.image.load('./saved/display/thermometer.png')
	mask = pygame.image.load('./saved/display/covid.png')
	tick = pygame.image.load('./saved/display/tick.png')
	cross = pygame.image.load('./saved/display/close.png')
	check = pygame.image.load('./saved/display/check.jpg')
	check = pygame.transform.scale(check,(300,300))
	# Title
	pygame.display.set_caption('CUV Technologies')
	    
	s = pygame.Surface((bbwidth,bbheight), pygame.SRCALPHA)   # per-pixel alpha
		                 # notice the alpha value in the color

	running = True
	while running:
		if all_ok:
			screen.fill((220,255,220))
			s.fill((143,255,143,128))
		else:
			screen.fill((255,220,220))
			s.fill((255,143,143,128))
		screen.blit(s, (bbx,bby))
		# pygame.draw.rect(screen,(0,255,120),(200,150,380,330),0)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					running = False
		screen.blit(mask,(maskx,masky))
		screen.blit(mask_sign(mask_detected,tick,cross),(mask_signx,mask_signy))
		text = font.render(show_mask_text(mask_detected),True,(255,255,255))
		screen.blit(text,(mtextx,mtexty))

		screen.blit(check,(checkx,checky))
		screen.blit(temperature,(tempx,tempy))
		screen.blit(temp_sign(temperature_normal,tick,cross),(temp_signx,temp_signy))
		text = font.render(show_temp_text(temperature_normal,temp,normal_temp),True,(255,255,255))
		screen.blit(text,(ttextx,ttexty))
		
		pygame.display.update()
	pygame.display.quit()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
pygame.init()
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		
		mask_detected = True if mask>withoutMask else False
		#void detect_temp()
		temperature_normal = True
		temp =37
		
		display(mask_detected,temperature_normal,temp)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
