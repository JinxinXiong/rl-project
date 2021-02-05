import tensorflow as tf
# print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tqdm


def mnist_train_test(number=2, mode='larger'):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	# Rescale the images from [0,255] to the [0.0,1.0] range.
	x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

	train_index = y_train==number
	test_index = y_test==number
	train = x_train[train_index]
	test = x_test[test_index]

	shuffle_train_index = np.arange(train.shape[0])
	shuffle_test_index = np.arange(test.shape[0])
	np.random.shuffle(shuffle_train_index)
	np.random.shuffle(shuffle_test_index)
	train_target = train[shuffle_train_index]
	test_target = test[shuffle_test_index]

	if mode == 'larger':
		train = np.pad(train[:,], [(0, 0), (8, 8), (8, 8), (0, 0)], 'constant')

	return train, train_target, test, test_target


def get_disturbed_data(x):
	'''
	x: batch_size, height, width, channel
	return: 
		zoom_in_x: fixed images (shrinked to able to test larger movemanet)
		pertub_x: moving images
		params: ground truth parameters, (theta: 1, tx: 1, ty: 1, zx: 0.05, zy: 0.05)
	'''
	zoom_in_x = np.empty(x.shape)
	for i in range(x.shape[0]):
		zoom_in_x[i] = tf.keras.preprocessing.image.apply_affine_transform(
								x[i], theta=0, tx=0, ty=0, shear=0, zx=1.25, zy=1.25, row_axis=0, col_axis=1,
								channel_axis=2, fill_mode='constant', cval=0.0, order=1)

	perturb_theta = np.random.randint(-3, 3, (x.shape[0], ))
	perturb_tx = np.random.randint(-6, 6, (x.shape[0], ))
	perturb_ty = np.random.randint(-6, 6, (x.shape[0], ))

	perturb_x = np.empty(x.shape)
	for i in range(x.shape[0]):
		perturb_x[i] = tf.keras.preprocessing.image.apply_affine_transform(
								x[i], 
								theta=perturb_theta[i]*30, 
								tx=perturb_tx[i], 
								ty=perturb_ty[i], shear=0, 
								zx=1.25, 
								zy=1.25, row_axis=0, col_axis=1,
								channel_axis=2, fill_mode='constant', cval=0.0, order=1)
	params = np.array(list(zip(perturb_theta, perturb_tx, perturb_ty)))

	return zoom_in_x.astype(np.float32), perturb_x.astype(np.float32), params

def get_disturbed_data_larger(x):
	'''
	x: batch_size, height, width, channel
	return: 
		zoom_in_x: fixed images (shrinked to able to test larger movemanet)
		pertub_x: moving images
		params: ground truth parameters, (theta: 1, tx: 1, ty: 1, zx: 0.05, zy: 0.05)
	'''
	zoom_in_x = np.empty(x.shape)
	for i in range(x.shape[0]):
		zoom_in_x[i] = tf.keras.preprocessing.image.apply_affine_transform(
								x[i], theta=0, tx=0, ty=0, shear=0, zx=1.25, zy=1.25, row_axis=0, col_axis=1,
								channel_axis=2, fill_mode='constant', cval=0.0, order=1)

	perturb_theta = np.random.randint(-4, 5, (x.shape[0], ))
	perturb_tx = np.random.randint(-12, 12, (x.shape[0], ))
	perturb_ty = np.random.randint(-12, 12, (x.shape[0], ))

	perturb_x = np.empty(x.shape)
	for i in range(x.shape[0]):
		temp = tf.keras.preprocessing.image.apply_affine_transform(
								x[i], 
								theta=perturb_theta[i]*30, 
								tx=0, 
								ty=0, shear=0, 
								zx=1, 
								zy=1, row_axis=0, col_axis=1,
								channel_axis=2, fill_mode='constant', cval=0.0, order=1)
		perturb_x[i] = tf.keras.preprocessing.image.apply_affine_transform(
								temp, 
								theta=0, 
								tx=perturb_tx[i], 
								ty=perturb_ty[i], shear=0, 
								zx=1.25, 
								zy=1.25, row_axis=0, col_axis=1,
								channel_axis=2, fill_mode='constant', cval=0.0, order=1)

	params = np.array(list(zip(perturb_theta, perturb_tx, perturb_ty)))

	return zoom_in_x.astype(np.float32), perturb_x.astype(np.float32), params

def generate_batch(x, batch_size=1, mode='larger'):
	curr_start_index = 0
	idx_list = np.arange(len(x))
	while True:
		x_img_list = []
		curr_end_idx = curr_start_index + batch_size

		while curr_end_idx > len(idx_list):
			idx_list = np.append(idx_list, idx_list)
		for idx in idx_list[curr_start_index: curr_end_idx]:
			x_img_list.append(x[idx])
		curr_start_index += batch_size

		if curr_start_index > len(x):
			curr_start_index = 0
			idx_list = np.arange(len(x))

		x_img_list = np.array(x_img_list, dtype=np.float32)

		if mode=='larger':
			fixed, moving, params = get_disturbed_data_larger(x_img_list)
		else:
			fixed, moving, params = get_disturbed_data(x_img_list)

		yield fixed, moving, params

def main():
	# mode = 'larger'
	train, _, _, _ = mnist_train_test(2)
	batch_generator = generate_batch(train[1000:2020])
	fixed, moving, params = next(batch_generator)
	print(fixed.shape)


if __name__ == '__main__':
	main()