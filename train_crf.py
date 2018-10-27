import pycrfsuite as pcrf
import os
import cv2
import pickle
import numpy as np
import crf


image_files = os.listdir(os.getcwd()+'/images')
train_images = [cv2.imread('images/'+filename, 0)for filename in image_files]
label_images = [cv2.imread('labels/'+filename, 0) for filename in image_files]
segmt_images = [pickle.load(open('segs/'+filename[:-4]+'.pkl','r')) for filename in image_files]
x_train = [crf.img2features(img, seg) for img, seg in zip(train_images, segmt_images)]
y_train = [crf.img2labels(img) for img in label_images]

print "2"
trainer = pcrf.Trainer(verbose=False)

for xseq, yseq in zip(x_train, y_train):
	trainer.append(xseq, yseq)

trainer.set_params({
	'c1' : 1.0,
	'c2' : 1e-3,
	'max_iterations' : 10,
	'feature.possible_transitions':True
	})

trainer.train('building_area.crfsuite')



tests = os.listdir(os.getcwd()+'/tests')
test_images = [cv2.imread('tests/'+filename, 0)for filename in tests]

test_segs = [pickle.load(open('test_segs/'+filename[:-4]+'.pkl','r')) for filename in tests]

x_test = [crf.img2features(img, seg) for img, seg in zip(test_images, test_segs)]

tagger = pcrf.Tagger()
tagger.open('building_area.crfsuite')

'''for tests in x_test:
	for test in tests:
		print type(test)'''

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