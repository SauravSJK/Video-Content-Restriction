import argparse
import numpy as np
import tensorflow as tf 
import keras, os, subprocess
import keras.backend as K
from keras import initializers
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, concatenate

imgsize = 100
numoffiles = 21
bdir1 = 'Age_Gender_Files/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
CHECKPOINT_FILE_PATH_AGE = 'age_model.h5'
CHECKPOINT_FILE_PATH_GENDER = 'gender_model.h5'
gclass_names = { 0 : 'm', 1 : 'f'}
aclass_names = { 0 : '(0,18)', 1 : '(19,100)' }
training_validation_filenames, testing_filenames = ([] for i in range(2))
"""bdir1 = '/mnt/c/Users/saura/Documents/Adience/Age_Gender_Files/'"""

def get_filenames():
	for i in range(numoffiles):
		training_validation_filenames.append(bdir1 + 'Age_Gender_' + str(i) + '.tfrecords')
	#testing_filenames.append(bdir2 + 'Test.tfrecords')

def extract_fntv(data_record):
	features = { 'Age' : tf.FixedLenFeature([], tf.int64), 'Gender' : tf.FixedLenFeature([], tf.int64), 'Image_raw' : tf.FixedLenFeature([], tf.string), 'Feature_set' : tf.VarLenFeature(tf.float32) }
	sample = tf.parse_single_example(data_record, features)
	Age = sample['Age']
	Gender = sample['Gender']
	Image = tf.decode_raw(sample['Image_raw'], tf.uint8)
	Image_shape = tf.stack([imgsize, imgsize, -1])
	Image = tf.reshape(Image, Image_shape)
	Feature_set = sample['Feature_set']
	return [Age, Gender, Image, Feature_set]

def training_validation_input():
	training_validation_dataset = tf.data.TFRecordDataset(training_validation_filenames)
	training_validation_dataset = training_validation_dataset.map(extract_fntv)
	iterator = training_validation_dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	temp_img, atemp_label, gtemp_label, gabor = ([] for i in range(4))
	with tf.Session() as sess:
		try:
			while True:
				data_record = sess.run(next_element)
				temp_img.append(data_record[2].astype('float32'))
				gtemp_label.append(data_record[1])
				atemp_label.append(data_record[0])
				gabor.append(data_record[3][1])
		except:
			pass
	return (np.stack(temp_img) / 255.0), to_categorical(np.stack(atemp_label)), to_categorical(np.stack(gtemp_label)), np.stack(gabor)

def train_network(train_gabor, train_img, atrain_class, gtrain_class, args):
	os.system("rm *model*")
	initializer = initializers.RandomNormal(mean = 0.0, stddev = 0.01, seed = None) 
	adam = Adam(lr = 0.001, decay = 0.0)
	es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 200)
	
	inputA = Input(shape = (24, ))
	x = Dense(units = 256, activation = 'relu', kernel_initializer = initializer, input_shape = (24, ))(inputA)
	inputB = Input(shape = (imgsize, imgsize, 3))
	y = Conv2D(96, (7, 7), activation = 'relu', kernel_initializer = initializer, input_shape = (imgsize, imgsize, 3))(inputB)
	y = MaxPooling2D(pool_size = (3, 3), strides = 2)(y)
	y = Conv2D(256, (5, 5), activation = 'relu', kernel_initializer = initializer)(y)
	y = MaxPooling2D(pool_size = (3, 3), strides = 2)(y)
	y = Conv2D(384, (3, 3), activation = 'relu', kernel_initializer = initializer)(y)
	y = MaxPooling2D(pool_size = (3, 3), strides = 2)(y)
	y = Flatten()(y)
	y = Dense(units = 256, activation = 'relu', kernel_initializer = initializer)(y)
	z = concatenate([x, y])
	z = Dropout(0.5)(z)
	z = Dense(units = 512, activation = 'relu', kernel_initializer = initializer)(z)
	z = Dropout(0.5)(z)
	a = Dense(units = 2, activation = 'softmax',kernel_initializer = initializer)(z)
	b = Dense(units = 2, activation = 'softmax',kernel_initializer = initializer)(z)
	modela = Model(inputs = [inputA, inputB], outputs = a)
	modelg = Model(inputs = [inputA, inputB], outputs = b)
	modela.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	modelg.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	
	checkpoint_path = CHECKPOINT_FILE_PATH_AGE
	checkpoint_path = os.path.join(args.job_dir, checkpoint_path)
	mc = ModelCheckpoint(checkpoint_path, monitor = 'val_acc', verbose = 1, save_best_only = True, period = 1, mode = 'max')
	modela.fit([train_gabor, train_img], atrain_class, epochs = 1000, batch_size = 500, verbose = 1, validation_split = 0.3, callbacks = [es, mc])
	modela.summary()
	
	checkpoint_path = CHECKPOINT_FILE_PATH_GENDER
	checkpoint_path = os.path.join(args.job_dir, checkpoint_path)
	mc = ModelCheckpoint(checkpoint_path, monitor = 'val_acc', verbose = 1, save_best_only = True, period = 1, mode = 'max')
	modelg.fit([train_gabor, train_img], gtrain_class, epochs = 1000, batch_size = 500, verbose = 1, validation_split = 0.3, callbacks = [es, mc])
	modelg.summary()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--job-dir', type=str, help='GCS or local dir to write checkpoints and export model', default='')
	args, _ = parser.parse_known_args()
	get_filenames()
	train_img, atrain_class, gtrain_class, train_gabor = training_validation_input()
	#test_img, atest_class, gtest_class, test_gabor = testing_input()
	train_network(train_gabor, train_img, atrain_class, gtrain_class, args)
	#test_network(test_img, atest_class, gtest_class, test_gabor)