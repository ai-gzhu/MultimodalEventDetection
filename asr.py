
import argparse
import torch
import torchvision
import speech_recognition as sr
import wave
import math
import struct
import sys

AUDIO_FILE = "./video-audio.wav"

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the input video")
args = vars(ap.parse_args())

vframes, aframes, info = torchvision.io.read_video(args["video"], pts_unit='sec')

aframes = torch.div(torch.sum(aframes, dim=0), len(aframes))
aframes = torch.add(torch.mul(aframes, .5), .5)
aframes = torch.mul(aframes, 255)
aframes = aframes.char()


aframes = aframes.numpy()

frames_processed_per_second = info["audio_fps"]*3

for i in range(0, math.floor(len(aframes)/frames_processed_per_second)):

 wave_writer = wave.open(AUDIO_FILE, 'w')

 wave_writer.setnchannels(1) # mono
 wave_writer.setsampwidth(1)
 wave_writer.setframerate(info["audio_fps"])
 
 wave_writer.writeframesraw(bytes(aframes[i*frames_processed_per_second:(i+1)*frames_processed_per_second]))
  
 wave_writer.close()

 r = sr.Recognizer()

 h = None

 # recognize speech using Sphinx
 try:
  with sr.AudioFile(AUDIO_FILE) as source:
   audio = r.record(source)  # read the entire audio file
   h = r.recognize_sphinx(audio,show_all=True,keyword_entries=[("grandfather", 0.99), ("cat", 0.99)])
   for s in h.seg():
    print(s.word, s.prob)#'| %4ss | %4ss | %8s |' % (s.start_frame / info["audio_fps"], s.end_frame / info["audio_fps"], s.word))
 except KeyboardInterrupt:
  exit()
 except:
  e = sys.exc_info()[0]
  print("Sphinx error; {0}".format(e))