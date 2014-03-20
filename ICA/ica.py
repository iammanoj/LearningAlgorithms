# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:32:48 2013

@author: sdey
"""

import math
import wave
import struct
import numpy as np

mix_file = "/Users/sdey/Documents/cs229/Assignments/ps4/q4/mix.dat"

mat = np.loadtxt(mix_file)

freq = 440.0
data_size = 40000
fname = "/Users/sdey/Documents/cs229/Assignments/ps4/q4/mix1.wav"
frate = 11025.0 # framerate as a float
amp = 64000.0     # multiplier for amplitude

sine_list_x = []
for x in range(data_size):
    sine_list_x.append(math.sin(2*math.pi*freq*(x/frate)))

wav_file = wave.open(fname, "w")

nchannels = 1
sampwidth = 2
framerate = int(frate)
nframes = data_size
comptype = "NONE"
compname = "not compressed"

wav_file.setparams((nchannels, sampwidth, framerate, nframes,
    comptype, compname))

for s in sine_list_x:
    # write the audio frames to file
    wav_file.writeframes(struct.pack('h', int(s*amp/2)))

wav_file.close()