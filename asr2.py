
import argparse
import torch
import torchvision
import speech_recognition as sr
from speech_recognition import AudioData, PortableNamedTemporaryFile
import wave
import math
import struct
import sys
import os
import io

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

frames_processed_per_second = math.floor(len(aframes))#info["audio_fps"]*3

def recognize_sphinx(audio_data, keyword_entries=None, grammar=None):
    """
    Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using CMU Sphinx.
    The recognition language is determined by ``language``, an RFC5646 language tag like ``"en-US"`` or ``"en-GB"``, defaulting to US English. Out of the box, only ``en-US`` is supported. See `Notes on using `PocketSphinx <https://github.com/Uberi/speech_recognition/blob/master/reference/pocketsphinx.rst>`__ for information about installing other languages. This document is also included under ``reference/pocketsphinx.rst``. The ``language`` parameter can also be a tuple of filesystem paths, of the form ``(acoustic_parameters_directory, language_model_file, phoneme_dictionary_file)`` - this allows you to load arbitrary Sphinx models.
    If specified, the keywords to search for are determined by ``keyword_entries``, an iterable of tuples of the form ``(keyword, sensitivity)``, where ``keyword`` is a phrase, and ``sensitivity`` is how sensitive to this phrase the recognizer should be, on a scale of 0 (very insensitive, more false negatives) to 1 (very sensitive, more false positives) inclusive. If not specified or ``None``, no keywords are used and Sphinx will simply transcribe whatever words it recognizes. Specifying ``keyword_entries`` is more accurate than just looking for those same keywords in non-keyword-based transcriptions, because Sphinx knows specifically what sounds to look for.
    Sphinx can also handle FSG or JSGF grammars. The parameter ``grammar`` expects a path to the grammar file. Note that if a JSGF grammar is passed, an FSG grammar will be created at the same location to speed up execution in the next run. If ``keyword_entries`` are passed, content of ``grammar`` will be ignored.
    Returns the most likely transcription if ``show_all`` is false (the default). Otherwise, returns the Sphinx ``pocketsphinx.pocketsphinx.Decoder`` object resulting from the recognition.
    Raises a ``speech_recognition.UnknownValueError`` exception if the speech is unintelligible. Raises a ``speech_recognition.RequestError`` exception if there are any issues with the Sphinx installation.
    """
    assert isinstance(audio_data, AudioData), "``audio_data`` must be audio data"
    assert keyword_entries is None or all(isinstance(keyword, (type(""), type(u""))) and 0 <= sensitivity <= 1 for keyword, sensitivity in keyword_entries), "``keyword_entries`` must be ``None`` or a list of pairs of strings and numbers between 0 and 1"
    # import the PocketSphinx speech recognition module
    try:
     from pocketsphinx import pocketsphinx, Jsgf, FsgModel, get_model_path, get_data_path

    except ImportError:
     raise RequestError("missing PocketSphinx module: ensure that PocketSphinx is set up correctly.")
    except ValueError:
     raise RequestError("bad PocketSphinx installation; try reinstalling PocketSphinx version 0.0.9 or better.")
    if not hasattr(pocketsphinx, "Decoder") or not hasattr(pocketsphinx.Decoder, "default_config"):
     raise RequestError("outdated PocketSphinx installation; ensure you have PocketSphinx version 0.0.9 or better.")
     

    # create decoder object
    

    model_path = get_model_path()
    data_path = get_data_path()



    # Create a decoder with certain model

    config = pocketsphinx.Decoder.default_config()
    config.set_string('-hmm', os.path.join(model_path, 'en-us'),)
    config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
    config.set_string('-dict', os.path.join(model_path, 'cmudict-en-us.dict'))
    config.set_string("-logfn", os.devnull)  # disable logging (logging causes unwanted output in terminal)
    decoder = pocketsphinx.Decoder(config)
    
    # obtain audio data
    raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)  # the included language models require audio to be 16-bit mono 16 kHz in little-endian format


    # obtain recognition results
    if keyword_entries is not None:  # explicitly specified set of keywords
     with PortableNamedTemporaryFile("w") as f:
      # generate a keywords file - Sphinx documentation recommendeds sensitivities between 1e-50 and 1e-5
      f.writelines("{} /1e{}/\n".format(keyword, 100 * sensitivity - 110) for keyword, sensitivity in keyword_entries)
      f.flush()
      
      # perform the speech recognition with the keywords file (this is inside the context manager so the file isn;t deleted until we're done)
      decoder.set_kws("keywords", f.name)
      decoder.set_search("keywords")
    elif grammar is not None:  # a path to a FSG or JSGF grammar
     if not os.path.exists(grammar):
      raise ValueError("Grammar '{0}' does not exist.".format(grammar))
     grammar_path = os.path.abspath(os.path.dirname(grammar))
     grammar_name = os.path.splitext(os.path.basename(grammar))[0]
     fsg_path = "{0}/{1}.fsg".format(grammar_path, grammar_name)
     if not os.path.exists(fsg_path):  # create FSG grammar if not available
      jsgf = Jsgf(grammar)
      rule = jsgf.get_rule("{0}.{0}".format(grammar_name))
      fsg = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
      fsg.writefile(fsg_path)
     else:
      fsg = FsgModel(fsg_path, decoder.get_logmath(), 7.5)
      decoder.set_fsg(grammar_name, fsg)
      decoder.set_search(grammar_name)

    decoder.start_utt()  # begin utterance processing
    decoder.process_raw(raw_data, False, True)  # process audio data with recognition enabled (no_search = False), as a full utterance (full_utt = True)
    decoder.end_utt()  # stop utterance processing

    return decoder

def remove_meta(word):
  start = word.find( '<' )
  end = word.find( '>' )
  if start != -1 and end != -1:
    word = word[0:start]+word[end+1: len(word)]
    
  start = word.find( '(' )
  end = word.find( ')' )
  if start != -1 and end != -1:
    word = word[0:start]+word[end+1: len(word)]

  start = word.find( '[' )
  end = word.find( ']' )
  if start != -1 and end != -1:
    word = word[0:start]+word[end+1: len(word)]
  return word

for i in range(0, math.floor(len(aframes)/frames_processed_per_second)):

    byte_memory = io.BytesIO()
    wave_writer = wave.open(byte_memory, 'w')

    wave_writer.setnchannels(1) # mono
    wave_writer.setsampwidth(1)
    wave_writer.setframerate(info["audio_fps"])
    
    wave_writer.writeframesraw(bytes(aframes[i*frames_processed_per_second:(i+1)*frames_processed_per_second]))
     
    wave_writer.close()

    r = sr.Recognizer()

    h = None

    byte_memory.seek(0)
    # recognize speech using Sphinx
    with sr.AudioFile(byte_memory) as source:
     audio = r.record(source)  # read the entire audio file
     h = recognize_sphinx(audio)#,keyword_entries=[("grandfather", 0.99), ("cat", 0.99)])
     for s in h.seg():
      result = s.word
      result = remove_meta(result)
      prob = math.pow(1.0001, s.prob)
      if result and prob > .1:
        print('| {:4.2f} | {:4.2f} | {:16s} | {:.2} |' .format (s.start_frame/100., s.end_frame/100., result, prob))