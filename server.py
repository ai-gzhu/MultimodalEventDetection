"""
"""

#ResNet152 Image Classification and Manipulation
import torch
import torch.nn.functional as F
import torchvision
import imp
import io

#Word2Vec
from gensim.models import KeyedVectors
from gensim import matutils
from numpy import dot

#WebSocket Server
import os
import asyncio
import time
import random
import websockets
import posixpath
import mimetypes
import base64
from http import HTTPStatus
from messages_pb2 import Message
import threading

#Debugging
import cv2
import numpy as np
import math

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

print("Loading word2vec vectors")
w2v_model = KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)


loop = asyncio.get_event_loop()

class WebSocketServerProtocolWithHTTP(websockets.WebSocketServerProtocol):
    """Implements a simple static file server for WebSocketServer"""

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def process_request(self, path, request_headers):
        """Serves a file when doing a GET request with a valid path"""
        self.max_size = 2**20

        if "Upgrade" in request_headers:
            return  # Probably a WebSocket connection

        if path == '/':
            path = '/index.html'

        response_headers = [
            ('Server', 'asyncio'),
            ('Connection', 'close'),
        ]
        server_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"www")
        full_path = os.path.realpath(os.path.join(server_root, path[1:]))

        print("GET", path, end=' ')

        # Validate the path
        if os.path.commonpath((server_root, full_path)) != server_root or \
                not os.path.exists(full_path) or not os.path.isfile(full_path):
            print("404 NOT FOUND")
            return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'

        print("200 OK")
        body = open(full_path, 'rb').read()
        ctype = self.guess_type(path)
        response_headers.append(("Content-type", ctype))
        response_headers.append(('Content-Length', str(len(body))))
        return HTTPStatus.OK, response_headers, body

    def guess_type(self, path):
        """Guess the type of a file.

        Argument is a PATH (a filename).

        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.

        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.

        """
 
        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']
 
    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
        })

def process_upload(websocket, proto):
    """
    print(proto.type)
    print(proto.extension)
    print(proto.keywords)
    print(proto.useImages)
    print(proto.useSounds)
    print(proto.imageChangeThreshold)
    print(proto.queryWindowDuration)
    print(proto.progress)
    print(proto.error)
    """
    
    random_vec = w2v_model.word_vec("cat")
    num_features = len(random_vec)
    feature_vector = np.zeros((num_features,),dtype=random_vec.dtype)

    word_count = 0
    for word in proto.keywords:
        try:
            word_vec = w2v_model.word_vec(word)
            feature_vector = np.add(feature_vector, word_vec)
            word_count = word_count + 1.
        except:
            pass
    
    if word_count:
        feature_vector = np.divide(feature_vector, word_count)

    server_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"www")
    videos_root = os.path.join(server_root,"videos")
    time_stamp = time.time_ns()
    short_upload_name = "upload.{}{}".format(time_stamp, proto.extension)
    short_download_name = "download.{}{}".format(time_stamp, proto.extension)
    uploaded_file_name = os.path.join(videos_root, short_upload_name)
    download_file_name = os.path.join(videos_root, short_download_name)
    if not proto.useImages and not proto.useSounds:
        message = Message()
        message.type = Message.ERROR
        message.error = "You must select at least one mode of analysis (images and/or sounds)."
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
    elif not word_count:
        message = Message()
        message.type = Message.ERROR
        message.error = "The specified keywords create a null feature vector."
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
    elif os.path.isfile(uploaded_file_name):
        message = Message()
        message.type = Message.ERROR
        message.error = "The uploaded file already exists on the server."
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
    elif os.path.commonpath((videos_root, uploaded_file_name)) != videos_root:
        message = Message()
        message.type = Message.ERROR
        message.error = "Invalid file extension."
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
    else:
        message = Message()
        message.type = Message.PROGRESS
        message.error = "Saving input file \"videos/"+short_upload_name+"\" to disk."
        message.progress = 0
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
        
        print("Saving uploaded video as: {}".format(uploaded_file_name))
        with open(uploaded_file_name, "wb") as new_file:
            new_file.write(proto.video)
            new_file.close()
                
        vframes, aframes, info = torchvision.io.read_video(uploaded_file_name, pts_unit='sec')

        original_vframes = vframes.clone()

        vframes = vframes.to(device)
            
        last_frame = None
        last_weight = 0

        update_frequency = math.floor(len(vframes)*.05)
        current_frame = 0

        weights = []

        for vframe in vframes:

            if current_frame % update_frequency == 0:
                message = Message()
                message.type = Message.PROGRESS
                message.error = "Processing..."
                message.progress = current_frame*1.0/len(vframes)
                asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)

            orig = vframe
            vframe = vframe.float()
            vframe = torch.transpose(vframe, 1, 2)
            vframe = torch.transpose(vframe, 0, 1)
            vframe = F.interpolate(vframe, size=224)  #The resize operation on tensor.
            
            skip = False
            if type(last_frame) != type(None):
                if (last_frame-vframe).abs().sum()/224./224./256. < proto.imageChangeThreshold/100.0:
                    skip = True
                else:
                    last_frame = vframe
            else:        
                last_frame = vframe
            
            if not skip:
                vframe = vframe.unsqueeze(0)

                output = model(vframe)

                h_x = output.data.squeeze()
                probs, idx = h_x.sort(0, True)

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
                            image_word_vector = w2v_model.get_vector(tokens1[s1])
                            s = dot(matutils.unitvec(feature_vector), matutils.unitvec(image_word_vector))
                            #w2v_model.similarity(feature_vector, tokens1[s1])
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
                last_weight = similarity/total
                """
                barGraph = ""
                numBars = 100
                barRatio = similarity/total#math.sqrt(top_combination_value)
                
                for i in range(0, math.floor(numBars*barRatio)):
                    barGraph += "|"
                for i in range(math.floor(numBars*barRatio), numBars):
                    barGraph += "-"
            
                print(barGraph)
                               
                #print('Top Recognition was {:.2f}% for {}'.format(top_probability_value* 100.0, top_probability_string))
                #print('Top Similarity was {:.2f}% for {}'.format(top_similarity_value * 100.0, top_similarity_string))
                #print('Top Combination was {:.2f}% for {}'.format(math.sqrt(top_combination_value) * 100.0, top_combination_string))
                
                #print('Top-N Results: ')
                #for i in range(0, num_predictions):
                #	print('{:.2f}% -> {}'.format(probs[i] * 100.0, labels[idx[i]]))
                """
                """
                cv2.imshow("Classification", cv2.cvtColor(np.array((orig.cpu())), cv2.COLOR_BGR2RGB))

                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                """
            current_frame += 1
            weights.append(last_weight)

        message = Message()
        message.type = Message.PROGRESS
        message.error = "Saving output file \"videos/"+short_download_name+"\" to disk."
        message.progress = 0
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)
            
        weights = torch.Tensor(weights)

        descending_weights, sorted_index = torch.sort(weights, dim=0, descending=True)

        rearranged_frames = original_vframes.index_select(0, sorted_index)

        torchvision.io.write_video(download_file_name, rearranged_frames, int(.5+info["video_fps"]))
        
        message = Message()
        message.type = Message.RESULT
        message.extension = short_download_name
        asyncio.run_coroutine_threadsafe(websocket.send(message.SerializeToString()), loop=loop)


async def on_connection(websocket, path):
    while True:
        #now = datetime.datetime.utcnow().isoformat() + 'Z'
        #await websocket.send(now)
        length = int(await websocket.recv())
        array = b''
        while len(array) != length:
            array += await websocket.recv()
            message = Message()
            message.type = Message.PROGRESS
            message.error = "Uploading..."
            message.progress = len(array)*1.0/length
            await websocket.send(message.SerializeToString())
        message = Message()
        threading.Thread(target=process_upload, args=(websocket, message.FromString(array))).start()


if __name__ == "__main__":
    start_server = websockets.serve(on_connection, 'localhost', 80,
                                    create_protocol=WebSocketServerProtocolWithHTTP)
    print("Running server at http://localhost:80/")

    loop.run_until_complete(start_server)
    loop.run_forever()