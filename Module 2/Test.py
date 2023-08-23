import keras, cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.models import load_model
from prettytable import PrettyTable

imgsize = 100
gclass = {'m' : 0, 'f' : 1}
gclass_names = { 0 : 'm', 1 : 'f'}
aclass_names = { 0 : '(0,18)', 1 : '(19,100)' }
bdir2 = '/mnt/c/Users/saura/Documents/Adience/Test_Images/Images/'
files = [f for f in listdir(bdir2) if isfile(join(bdir2, f))]
temp_img, gtemp_label, atemp_label, gabor, filters = ([] for i in range(5))
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

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

def build_test_data():
	global temp_img, gabor
	for file in files:
		img = cv2.imread(bdir2 + file)
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
			except Exception as e:
				pass
		index = file.strip().split('_')
		atemp_label.append(int(index[1]))
		gtemp_label.append(int(index[2]))
		temp_img.append(img)
		gabor.append(feature_set)
	temp_img = (np.stack(temp_img) / 255.0)
	gabor = np.stack(gabor)

def predict():
	a_acc, g_acc = (0.0 for i in range(2))
	model = load_model('age_model.h5')
	model.summary()
	predictions = model.predict([gabor, temp_img])
	t = PrettyTable(['Predicted Age', 'Actual Age', 'Filename'])
	for i in range(len(predictions)):
		t.add_row([aclass_names[np.argmax(predictions[i])], aclass_names[atemp_label[i]], files[i]])
		if np.argmax(predictions[i]) == atemp_label[i]:
			a_acc += 1
	print t
	model = load_model('gender_model.h5')
	predictions = model.predict([gabor, temp_img])
	t = PrettyTable(['Predicted Gender', 'Actual Gender', 'Filename'])
	for i in range(len(predictions)):
		t.add_row([gclass_names[np.argmax(predictions[i])], gclass_names[gtemp_label[i]], files[i]])
		if np.argmax(predictions[i]) == gtemp_label[i]:
			g_acc += 1
	print t
	print "Age Accuracy : ", (a_acc / len(predictions))
	print "Gender Accuracy : ", (g_acc / len(predictions))

build_filters()
build_test_data()
predict()