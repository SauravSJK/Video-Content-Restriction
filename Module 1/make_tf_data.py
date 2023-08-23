import cv2
import tensorflow as tf 
import numpy as np
import os, shutil
from os import listdir
from os.path import isfile, join

imgsize = 100
numfile = 2000
EPS = 0.00000000000000001
agdata, filters = ([] for i in range(2))
gclass_names = { 0 : 'm', 1 : 'f'}
gclass = {'m' : 0, 'f' : 1}
aclass = {'(0,18)' : 0 , '(19,100)' : 1}
inaclass = ['(0, 2)', '(4, 6)', '(8, 12)', '(8, 23)', '(15, 20)', '(25, 32)', '(27, 32)', '(38, 42)', '(38, 43)', '(38, 48)', '(48, 53)', '(60, 100)']
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
bdir1 = ''
bdir2 = 'UTKFace/'
"""bdir1 = '/mnt/c/Users/saura/Documents/Adience/'
bdir2 = '/mnt/c/Users/saura/Documents/Adience/UTKFace/'"""
flist = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt']

def parse_input():
	global agdata
	alldata = []
	for txt in flist:
		with open(bdir1 + txt,'r') as f:
			lines = f.readlines()[1:]
			for line in lines:
				data = line.strip().split('\t')
				alldata.append([bdir1 + 'aligned/' + data[0] + '/landmark_aligned_face.' + data[2] + '.' + data[1], data[3], data[4]])
	files = [f for f in listdir(bdir2) if isfile(join(bdir2, f))]
	for file in files:
		index = file.strip().split('_')
		alldata.append([bdir2 + file, index[0], gclass_names[int(index[1])]])
	for data in alldata:
		try:
			if data[1] == '(8, 12)' or data[1] == '(4, 6)' or data[1] == '(8, 23)' or data[1] == '(0, 2)':
				data[1] = '(0,18)'
			elif data[1] not in inaclass and data[1] != 'None' and int(data[1]) < 19:
				data[1] = '(0,18)'
			elif data[1] != 'None':
				data[1] = '(19,100)'
			if data[2] != '' and data[1] != 'None' and data[2] != 'u':
				agdata.append((data[0], aclass[data[1]], gclass[data[2]]))
		except Exception as e:
			print e

def create_base_directory():
	if os.path.exists(bdir1 + 'Age_Gender_Files'):
		shutil.rmtree(bdir1 + 'Age_Gender_Files')
	os.makedirs(bdir1 + 'Age_Gender_Files')

def build_filters():
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi/6):
		for psi in np.arange(0, 2*np.pi, np.pi):
			kern = cv2.getGaborKernel((ksize, ksize), 2.0, theta, 2.0, 0.3, psi, ktype=cv2.CV_32F)
			filters.append(kern)

def gfilter(img):
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	responses = []
	for kern in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		responses.append(fimg)
	return responses

def get_local_energy (matrix):
	local_energy = 0.0
	for row in range (len(matrix)):
		for col in range(len(matrix[0])):
			val = int(matrix[row][col]) * int(matrix[row][col])
			local_energy = local_energy + val
	local_energy = local_energy / 650250000
	return EPS if local_energy == 0 else local_energy

def get_mean_amplitude (matrix):
	mean_amp = 0.0
	for row in range (len(matrix)):
		for col in range(len(matrix[0])):
			val = abs(int(matrix[row][col]))
			mean_amp = mean_amp + val
	mean_amp = mean_amp / 2550000
	return EPS if mean_amp == 0 else mean_amp	

def make_tvdata():
	count = 0
	num = 0
	while True:
		writerag = tf.python_io.TFRecordWriter(bdir1 + 'Age_Gender_Files/Age_Gender_' + str(num) + '.tfrecords')
		for j in range(numfile):
			if count == len(agdata):
				return
			img = cv2.imread(agdata[count][0])
			gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			for i, face in enumerate(cascade.detectMultiScale(gray_image, scaleFactor = 1.2, minNeighbors = 14, minSize = (imgsize, imgsize))):
				x, y, w, h = face
				img = img[y:y + h, x:x + w]
			if img is not None:
				try:
					img = cv2.resize(img, (imgsize, imgsize))
					local_energy_results, mean_amplitude_results = ([] for i in range(2))
					for matrix in gfilter(img):
						local_energy_results.append(get_local_energy(matrix))
						mean_amplitude_results.append(get_mean_amplitude(matrix))
					feature_set = local_energy_results + mean_amplitude_results
					imgraw = img.tostring()
					feature = {'Age' : tf.train.Feature(int64_list = tf.train.Int64List(value = [agdata[count][1]])), 'Gender': tf.train.Feature(int64_list = tf.train.Int64List(value = [agdata[count][2]])), 'Image_raw' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [imgraw])), 'Feature_set' : tf.train.Feature(float_list = tf.train.FloatList(value = feature_set))}
					example = tf.train.Example(features = tf.train.Features(feature = feature))
					writerag.write(example.SerializeToString())
					count += 1
				except Exception as e:
					pass
		print count
		print "Age_Gender_" + str(num) + " completed!!!"
		num += 1
		
parse_input()
print "Input Parsed"
create_base_directory()
print "Created Directory"
build_filters()
print "Created Filters"
make_tvdata()
print "Created Training and Validation Data"