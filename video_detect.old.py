# USAGE
# python test_imagenet.py --image images/dog_beagle.png

# import the necessary packages
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import ResNet152
import numpy as np
import argparse
import cv2
import math
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the input video")
args = vars(ap.parse_args())



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(args["video"])
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)


#width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
#height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

# Check if camera opened successfully
if (cap.isOpened()== False): 
		print("Error opening video stream or file")
 
# load the ResNet101V2 network pre-trained on the ImageNet dataset
print("[INFO] loading network...")
model = ResNet152(weights="imagenet11k",include_top=True)

model.summary();

# Read until video is completed
while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
				image = frame[0:299, 0:299]
				#image = cv2.resize(frame, (299, 299	))
				orig = image
				image = image_utils.img_to_array(image)
				
				# our image is now represented by a NumPy array of shape (224, 224, 3),
				# assuming TensorFlow "channels last" ordering of course, but we need
				# to expand the dimensions to be (1, 3, 224, 224) so we can pass it
				# through the network -- we'll also preprocess the image by subtracting
				# the mean RGB pixel intensity from the ImageNet dataset
				image = np.expand_dims(image, axis=0)
				image = preprocess_input(image)


				# classify the image
				print("[INFO] classifying image...")
				preds = model.predict(image)
				P = imagenet_decode_predictions(preds)
				# loop over the predictions and display the rank-5 predictions +
				# probabilities to our terminal
				for (i, (imagenetID, label, prob)) in enumerate(P[0]):
								print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

				# load the image via OpenCV, draw the top prediction on the image,
				# and display the image to our screen
				(imagenetID, label, prob) = P[0][0]
				#cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
				#				(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
				cv2.imshow("Classification", orig)
 
				# Press Q on keyboard to		exit
				if cv2.waitKey(25) & 0xFF == ord('q'):
						break
 
		# Break the loop
		else: 
				break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

 
