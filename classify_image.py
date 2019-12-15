import torch
import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing import image
import argparse
import io
import imp

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
	
# ************** Parameters:
num_predictions = 5  # Top-k Results
model_address = './resnet152Full.pth'  # for loading models
lexicon_address = './synset.txt'
test_image_address = args["image"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MainModel = imp.load_source('MainModel', "kit_imagenet.py")
# Load Converted Model:
with open('./resnet152Full.pth', 'rb') as f:
	buffer = io.BytesIO(f.read())
torch.load(buffer)
model = torch.load(model_address).to(device)
model.eval()


# Read Input Image and Apply Pre-process:
img = image.load_img(test_image_address, target_size=(224, 224))
x = image.img_to_array(img)
x = x[..., ::-1]  # transform image from RGB to BGR
x = np.transpose(x, (2, 0, 1))
x = np.expand_dims(x, 0).copy()
x = torch.from_numpy(x)
x = x.to(device)


# Load Full-ImageNet Dictionary (i.e., lexicon):
with open(lexicon_address, 'r') as f:
	labels = [l.rstrip() for l in f]


# Make prediction (forward pass):
with torch.no_grad():
	output = model(x)
max, argmax = output.data.squeeze().max(0)
class_id = argmax.item()
class_name = labels[class_id]


# Print the top-5 Results:
h_x = output.data.squeeze()
probs, idx = h_x.sort(0, True)
print('Top-5 Results: ')
for i in range(0, num_predictions):
	print('{:.2f}% -> {}'.format(probs[i] * 100.0, labels[idx[i]]))
str_final_label = 'The Image is a ' + class_name[10:] + '.'
print(str_final_label)
