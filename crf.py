# A CRF module created

import pycrfsuite as pcrf
import os
import cv2
import pickle
import numpy as np

def pixel2features(img, seg, i, j, h, w) :
	features = [
		'bias',
		'row=' + str(float(i)/float(h)),
		'col=' + str(float(j)/float(w)),
		'color=' + str(img[i,j]),
		'seg_num=' + str(seg[i,j]),
	]
	if i>0:
		features.extend([
			'-10:row=' + str(float(i-1)/float(h)),
			'-10:col=' + str(float(j)/float(w)),
			'-10:color=' + str(img[i-1,j]),
			'-10:seg_num=' + str(seg[i-1,j]),
		])
	else :
		features.append('L')
	if i<h-1:
		features.extend([
			'+10:row=' + str(float(i+1)/float(h)),
			'+10:col=' + str(float(j)/float(w)),
			'+10:color=' + str(img[i+1,j]),
			'+10:seg_num=' + str(seg[i+1,j]),
		])
	else :
		features.append('R')
	if j>0:
		features.extend([
			'0-1:row=' + str(float(i)/float(h)),
			'0-1:col=' + str(float(j-1)/float(w)),
			'0-1:color=' + str(img[i,j-1]),
			'0-1:seg_num=' + str(seg[i,j-1]),
		])
	else :
		features.append('T')
	if j<w-1:
		features.extend([
			'0+1:row=' + str(float(i)/float(h)),
			'0+1:col=' + str(float(j+1)/float(w)),
			'0+1:color=' + str(img[i,j+1]),
			'0+1:seg_num=' + str(seg[i,j+1]),
		])
	else :
		features.append('B')
	return features


def img2features(img, seg) :
	h,w = img.shape[:2]
	return [pixel2features(img, seg, i, j, h, w) for i in range(h) for j in range(w)]

def img2labels(img) :
	h,w = img.shape[:2]
	return [str(img[i,j]) for i in range(h) for j in range(w)]