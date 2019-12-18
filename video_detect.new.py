# USAGE
# python test_imagenet.py --image images/dog_beagle.png

# import the necessary packages
from keras.preprocessing import image as image_utils
from gensim.models import KeyedVectors
import torch
import io
import imp
import numpy as np
import argparse
import cv2
import math
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the input video")
ap.add_argument("-q", "--query", required=True,
	help="the event to detect")
args = vars(ap.parse_args())


# Load Converted Model:
num_predictions = 50
model_address = './resnet152Full.pth'
lexicon_address = './synset.txt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MainModel = imp.load_source('MainModel', "kit_imagenet.py")
with open('./resnet152Full.pth', 'rb') as f:
	buffer = io.BytesIO(f.read())
torch.load(buffer)
model = torch.load(model_address).to(device)
model.eval()


# Load Full-ImageNet Dictionary (i.e., lexicon):
with open(lexicon_address, 'r') as f:
	labels = [l.rstrip() for l in f]


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


print("Loading word2vec vectors")
w2v_model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)

# Read until video is completed
while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:
				orig = frame
				frame = cv2.resize(frame, (224, 224	))
				#img = image_utils.img_to_array(frame)
								
				# Read Input Image and Apply Pre-process:
				x = image_utils.img_to_array(frame)
				x = x[..., ::-1]  # transform image from RGB to BGR
				x = np.transpose(x, (2, 0, 1))
				x = np.expand_dims(x, 0).copy()
				x = torch.from_numpy(x)
				x = x.to(device)
				
				
				
				
				# Make prediction (forward pass):
				with torch.no_grad():
					output = model(x)
				
				
				# Print the top-5 Results:
				h_x = output.data.squeeze()
				probs, idx = h_x.sort(0, True)
				
				
				similarity = 0.
				total = 0.01
				
				for i in range(0, num_predictions):
					tokens1 = labels[idx[i]].split(", ")
					for s1 in range(1, len(tokens1)):
						tokens2 = tokens1[s1].split(" ")
						for s2 in range(0, len(tokens2)):
							try:
								s = w2v_model.similarity(args["query"], tokens2[s2])
								p = probs[i]
								
								
								
								similarity += s*p
								total += p
							except:
								pass
								#do nothing
							
				
				#print(similarity / total)
				
				barGraph = ""
				numBars = 100
				
				for i in range(0, math.floor(numBars*(similarity/total))):
					barGraph += "|"
				for i in range(math.floor(numBars*(similarity/total)), numBars):
					barGraph += "-"
				
				print(barGraph);
				
				#print('Top-5 Results: ')
				#for i in range(0, num_predictions):
				#	print('{:.2f}% -> {}'.format(probs[i] * 100.0, labels[idx[i]]))
				#str_final_label = 'The Image is a ' + class_name + '.'
				#print(str_final_label)
				
				
				
				
				
				
				
				cv2.imshow("Classification", orig)
 
				# Press Q on keyboard to		exit
				if cv2.waitKey(1) & 0xFF == ord('q'):
						break
 
		# Break the loop
		else: 
				break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

 
