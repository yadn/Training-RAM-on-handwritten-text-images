#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 03:28:08 2019

@author: eisd
"""
import cv2
from IAMDataLoader import DataLoader, Batch

batchSize=32
imgSize=(128, 32)
maxTextLen=32

train=1
validate=0

loader = DataLoader('./data/', batchSize, imgSize, maxTextLen)

# save characters of model for inference mode
#open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
# save words contained in dataset into file
#open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

# execute training or validation
#if train==1:
#	model = Model(loader.charList, decoderType)
#	train(model, loader)
#elif validate==1:
#	model = Model(loader.charList, decoderType, mustRestore=True)
#	validate(model, loader)