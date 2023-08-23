import tensorflow as tf
import keras, cv2
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify
imgsize = 100
app = Flask(__name__)

def build_filters():
	ksize = 31
	filters = []
	for theta in np.arange(0, np.pi, np.pi/6):
		for psi in np.arange(0, 2*np.pi, np.pi):
			kern = cv2.getGaborKernel((ksize, ksize), 2.0, theta, 2.0, 0.3, psi, ktype=cv2.CV_32F)
			filters.append(kern)
	return filters

def gfilter(img, filters):
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

def build_test_data(filters):
	gabor, temp_img = ([] for i in range(2))
	img = cv2.imread('Project/image.jpg')
	if img is not None:
		img = cv2.resize(img, (imgsize, imgsize))
		local_energy_results, mean_amplitude_results = ([] for i in range(2))
		for matrix in gfilter(img, filters):
			local_energy_results.append(get_local_energy(matrix))
			mean_amplitude_results.append(get_mean_amplitude(matrix))
		feature_set = local_energy_results + mean_amplitude_results
	temp_img.append(img)
	gabor.append(feature_set)
	temp_img = (np.stack(temp_img) / 255.0)
	gabor = np.stack(gabor)
	return gabor, temp_img

@app.route("/", methods = ['POST'])
def main():
	if request.method == 'POST':
		f = request.files['image']
		f.save('Project/image.jpg')
		filters = build_filters()
		gabor, temp_img = build_test_data(filters)
		keras.backend.clear_session() 
		model = load_model('Project/age_model.h5')
		predictions = model.predict([gabor, temp_img])
		age = np.argmax(predictions[0])
		model = load_model('Project/gender_model.h5')
		predictions = model.predict([gabor, temp_img])
		gender = np.argmax(predictions[0])
		tf.keras.backend.clear_session()
	return jsonify(age = age, gender = gender)