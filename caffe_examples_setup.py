from __future__ import print_function

# Specify where PyCaffe is
caffe_path = '../../caffe_gt/python'

# Specify where the perpared ILSVRC2012 data is
imagenet_data = '/media/scratch/ILSVRC2012/data/'

# Import Numpy and Caffe plus a few other utilities
import sys, os, math, random, time, urllib, gzip
import numpy as np
import pickle as cPickle
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.append(caffe_path)
import caffe
from caffe import layers as L
from caffe import params as P
from caffe import to_proto

# Switch Caffe to GPU Mode
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

# Count available GPUs
print(caffe.enumerate_devices(True))

# Select the GPU
caffe.set_device(0)
