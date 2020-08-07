from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
import re
from SamplePreprocessor import preprocess

inputFile = "charset_size=82.txt"
filename_pref_no_path = os.path.split(inputFile)[1]

def read_charset(filename, null_character=u'\u2591'):
	pattern = re.compile(r'(\d+)\t(.+)')
	charset = {}
	with open(filename) as f:
		for i, line in enumerate(f):
			m = pattern.match(line)
			if m is None:
				logging.warning('incorrect charset file. line #%d: %s', i, line)
				continue
			code = int(m.group(1))
			char = m.group(2)#.decode('utf-8')
			if char == '<nul>':
				char = null_character
			charset[char] = code
	return charset

def encode_utf8_string(text='abc', charset={'a':0, 'b':1, 'c':2}, length=5, null_char_id=82):
	char_ids_padded = []
	char_ids_unpadded = []
	for i in range(0,len(text)):
		char_ids_unpadded.append(charset[text[i]])
		char_ids_padded.append(charset[text[i]])
	for i in range(len(text),length):
		char_ids_padded.append(null_char_id)
	return char_ids_padded,char_ids_unpadded

class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

	def __init__(self, filePath, batchSize, imgSize, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []
		self.charset = read_charset("charset_size=82.txt")
	
		f=open(filePath+'words.txt')
		lines = []
		for line in f:
			lines.append(line)
		# split into training and validation set: 95% - 5%
		#splitIdx = int(0.6 * len(self.samples))
		#print("len(lines)",len(lines),maxTextLen)
		self.trainSamples = self.sortlines(lines[:80421],filePath,maxTextLen)#80421
		#print("self.trainSamples",self.trainSamples)
		self.validationSamples = self.sortlines(lines[80421:80421+16770],filePath,maxTextLen)
#        print "splitIdx"
		# put words into lists
		self.trainWords = [x.gtText for x in self.trainSamples]
		maxlen = 0
		for x in self.trainSamples:
			maxlen = max(maxlen,len(x.gtText))
		#print("maxlen", maxlen)
		self.validationWords = [x.gtText for x in self.validationSamples]

		# number of randomly chosen samples per epoch for training 
		self.numTrainSamplesPerEpoch = 80421 
		# start with train set
		self.trainSet()

		# list of all chars in dataset
		#self.charList = sorted(list(chars))

	def sortlines(self, lines,filePath,maxTextLen):
		print("len(lines1)",len(lines))
		chars = set()
		bad_samples = []
		bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		gtTextlens = []
		samples1 = []
		#lines = []
		for line in lines:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			
			# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

			# GT text are columns starting at 9
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
			chars = chars.union(set(list(gtText)))

			# check if image is not empty
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue

			# put sample into list
			if (bad_samples_reference[0] not in fileName) and (bad_samples_reference[1] not in fileName):
				gtTextlens.append(len(gtText))
				samples1.append(Sample(gtText, fileName))
		sort_order = sorted(range(len(gtTextlens)), key=lambda k: gtTextlens[k])
		#print(sort_order)
		#gtTextlens = [gtTextlens[i] for i in sort_order]
		#print(gtTextlens)
		samples = [samples1[i] for i in sort_order]
		# some images in the IAM dataset are known to be damaged, don't show warning for them
		if set(bad_samples) != set(bad_samples_reference):
			print("Warning, damaged images found:", bad_samples)
			print("Damaged images expected:", bad_samples_reference)
		return samples

	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text


	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		#random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	
	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples


	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)
		
		
	def getNext(self):
		"iterator"
		batchRange = list(range(self.currIdx, self.currIdx + self.batchSize))
		np.random.shuffle(batchRange)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		gtTextsIndexes = []
		for text in gtTexts:
			char_ids_padded, char_ids_unpadded = encode_utf8_string(text=text, charset = self.charset, length=21, null_char_id = 82)
			#print(char_ids_padded, text)
			gtTextsIndexes.append(char_ids_padded)
		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTextsIndexes, imgs)


