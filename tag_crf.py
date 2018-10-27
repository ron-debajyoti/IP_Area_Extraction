import pycrfsuite as pcrf
import os
import cv2
import pickle
import numpy as np
import crf 

tests = os.listdir(os.getcwd()+'/tests')
test_images = [cv2.imread('tests/'+filename, 0)for filename in tests]

test_segs = [pickle.load(open('test_segs/'+filename[:-4]+'.pkl','r')) for filename in tests]

x_test = [crf.img2features(img, seg) for img, seg in zip(test_images, test_segs)]

tagger = pcrf.Tagger()
tagger.open('building_area.crfsuite')

y_preds = [tagger.tag(test) for test in x_test]
for i in range(len(y_preds)) :
	h,w = test_images[i].shape
	res = np.zeros((h,w), dtype = np.uint8)
	for x in range(h):
		for y in range(w):
			res[x,y] = int(y_preds[i][x*w+y])
	cv2.imwrite('crf_results/'+tests[i], res)
	cv2.imshow("i", res)
	cv2.waitKey(0)
cv2.destroyAllWindows()