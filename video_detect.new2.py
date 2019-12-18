# USAGE
# python test_imagenet.py --image images/dog_beagle.png

# import the necessary packages
from gensim.models import KeyedVectors
import torch
import torchvision
import io
import imp
import numpy as np
import argparse
import math
import time

import torch.nn.functional as F
import cv2
from keras.preprocessing import image as image_utils
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the input video")
ap.add_argument("-q", "--query", required=True,
	help="the event to detect")
args = vars(ap.parse_args())


# Load Converted Model:
num_predictions = 10
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

print("Loading word2vec vectors")
w2v_model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)

print("Loading video file")
vframes, aframes, info = torchvision.io.read_video(args["video"], pts_unit='sec')

#print(vframes)

#vframes = resize_tensors(vframes, 224, 224)

#print(vframes)
vframes = vframes.to(device)
    

for vframe in vframes:

    start = time.time()
    
    orig = vframe
    vframe = vframe.float()
    vframe = torch.transpose(vframe, 1, 2)
    vframe = torch.transpose(vframe, 0, 1)
    vframe = F.interpolate(vframe, size=224)  #The resize operation on tensor.
    #print(vframe.size())
    #vframe = cv2.cvtColor(cv2.resize(np.array(vframe), (224, 224)), cv2.COLOR_BGR2RGB)
    # Read Input Image and Apply Pre-process:
    
    #x = image_utils.img_to_array(vframe)
    #print(x)
    #x = F.interpolate(x, size=(224, 224), mode='linear', align_corners=False)
    # transform image from RGB to BGR
    #permute = [2, 1, 0]
    #x = x[:, permute]
    #x = np.transpose(x, (2, 0, 1))
    vframe = vframe.unsqueeze(0)
    #x = torch.from_numpy(x)
        
    end = time.time()
    print(end - start)
    
    
    
    start = time.time()
    # Make prediction (forward pass):
    #with torch.no_grad():
    output = model(vframe)
    
    
    end = time.time()
    print(end - start)

    start = time.time()
    
    # Print the top-5 Results:
    h_x = output.data.squeeze()
    probs, idx = h_x.sort(0, True)
    
    end = time.time()
    print(end - start)
    
    start = time.time()
    
    similarity = 0.
    total = 0.01
    
    top_similarity_value = 0.
    top_similarity_string = ""
    top_probability_value = 0.
    top_probability_string = ""
    top_combination_value = 0.
    top_combination_string = ""
    
    for i in range(0, num_predictions):
        tokens1 = labels[idx[i]].split(", ")
        for s1 in range(1, len(tokens1)):    
            try:
                s = w2v_model.similarity(args["query"], tokens1[s1])
                p = probs[i]
                similarity += s*p
                total += p
                if p > top_probability_value:
                    top_probability_value = p
                    top_probability_string = tokens1[s1]
                    
                if s > top_similarity_value:
                    top_similarity_value = s
                    top_similarity_string = tokens1[s1]
                    
                if s*p > top_combination_value:
                    top_combination_value = s*p
                    top_combination_string = tokens1[s1]
            except:
                pass
                #do nothing
            
            #tokens2 = tokens1[s1].split(" ")
            #for s2 in range(0, len(tokens2)):
            #    try:
            #        s = w2v_model.similarity(args["query"], tokens2[s2])
            #        p = probs[i]
            #        similarity += s*p
            #        total += p
            #    except:
            #        pass
            #        #do nothing
                
    
    #print(similarity / total)
    
    barGraph = ""
    numBars = 100
    barRatio = similarity/total#math.sqrt(top_combination_value)
    
    for i in range(0, math.floor(numBars*barRatio)):
        barGraph += "|"
    for i in range(math.floor(numBars*barRatio), numBars):
        barGraph += "-"
   
    end = time.time()
    print(end - start)
    
    print(barGraph);
    
    #print('Top Recognition was {:.2f}% for {}'.format(top_probability_value* 100.0, top_probability_string))
    #print('Top Similarity was {:.2f}% for {}'.format(top_similarity_value * 100.0, top_similarity_string))
    #print('Top Combination was {:.2f}% for {}'.format(math.sqrt(top_combination_value) * 100.0, top_combination_string))
    
    #print('Top-N Results: ')
    #for i in range(0, num_predictions):
    #	print('{:.2f}% -> {}'.format(probs[i] * 100.0, labels[idx[i]]))

    cv2.imshow("Classification", cv2.cvtColor(np.array((orig.cpu())), cv2.COLOR_BGR2RGB))

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Closes all the frames
#cv2.destroyAllWindows()
				
 

 
