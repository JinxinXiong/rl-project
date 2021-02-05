import tensorflow as tf
# print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tqdm
from data import *


class environment:
    def __init__(self, img_fixed, img_perturb, ground_truth_params, steps_per_game=25, eps=2.0):
        self.steps_per_game = steps_per_game
        self.fixed = img_fixed
        self.original_moving = img_perturb
        self.moving = img_perturb
        self.pre_moving = img_perturb
        self.ground_truth_params = ground_truth_params
        self.params = tf.zeros((self.fixed.shape[0], 3), dtype=tf.float32)
        self.prev_params = tf.zeros((self.fixed.shape[0], 3), dtype=tf.float32)
        self.steps = 0
        self.eps = eps
        self.params_history = []
        self.memo = ''
        self.revist = False

    def _reward(self, method='image'):
        self.prev_moving = self.warp_transform(self.original_moving, self.prev_params)
        self.moving = self.warp_transform(self.original_moving, self.params)
        distance = tf.math.reduce_euclidean_norm(self.moving-self.fixed, axis=[1,2,3]) / tf.math.reduce_max(self.moving)

        if method=='image':
            return tf.math.subtract(tf.cast(tf.math.reduce_euclidean_norm(self.prev_moving-self.fixed, axis=[1,2,3]), tf.float32),
                                    tf.cast(tf.math.reduce_euclidean_norm(self.moving-self.fixed, axis=[1,2,3]), tf.float32))
        else:
            # penalize if moving out
            if self.revist:
                self.memo = 'revisit'
                return tf.convert_to_tensor(np.array([True]*self.fixed.shape[0])), -1000.0
            if (tf.reduce_sum(tf.cast(self.moving!=0, tf.int32)) < tf.reduce_sum(tf.cast(self.pre_moving!=0, tf.int32))):# or (self.params[0][0] < -6 or self.params[0][0] > 6):
                self.memo = 'translation out of bound'
                return tf.convert_to_tensor(np.array([True]*self.fixed.shape[0])), -1000.0
            elif self.params[0][0] < -6 or self.params[0][0] > 6:
                self.memo = 'rotation out of bound'
                return tf.convert_to_tensor(np.array([True]*self.fixed.shape[0])), -1000.0
            elif distance < self.eps:
                self.memo = 'solve'
                return tf.convert_to_tensor(np.array([True]*self.fixed.shape[0])), 1000.0
            else:
                self.memo = 'unsolved but abey the rules'
                return tf.convert_to_tensor(np.array([False]*self.fixed.shape[0])), tf.cast(-tf.math.reduce_euclidean_norm(self.moving-self.fixed, axis=[1,2,3]), tf.float32)
    
    def _warp_transform(self, image, params):
        self.steps += 1
        next_moving = np.empty(self.fixed.shape, dtype=np.float32)
        for i in range(self.fixed.shape[0]):
            temp = tf.keras.preprocessing.image.apply_affine_transform(
                                image[0], 
                                theta=0, 
                                tx=params[0][1], 
                                ty=params[0][2], shear=0, 
                                zx=1, 
                                zy=1, row_axis=0, col_axis=1,
                                channel_axis=2, fill_mode='constant', cval=0.0, order=1)
            next_moving[i] = tf.keras.preprocessing.image.apply_affine_transform(
                                temp, 
                                theta=params[0][0]*30, 
                                tx=0, 
                                ty=-0, shear=0, 
                                zx=1, 
                                zy=1, row_axis=0, col_axis=1,
                                channel_axis=2, fill_mode='constant', cval=0.0, order=1)
            
        return next_moving
    
    def warp_transform(self, image, params):
        return tf.numpy_function(self._warp_transform, [image, params], [tf.float32])

    def _take_action(self, action, num_actions=4):
        '''
        take action as a integer
        (theta, tx, ty, zx, zy)
        0: theat+1, 1: theta-1
        2: tx+1,    3: tx-1
        4: ty+1,    5: ty-1
        6: zx, zy+0.05
        7: zx, zy-0.05
        '''
        self.steps += 1
        params = np.array([[0.0, 0.0, 0.0] for _ in range(self.fixed.shape[0])], dtype=np.float32)
        params[:, 0] += np.where(action==0, 1.0, 0.0)
        params[:, 0] += np.where(action==1, -1.0, 0.0)
        params[:, 1] += np.where(action==2, 1.0, 0.0)
        params[:, 1] += np.where(action==3, -1.0, 0.0)
        params[:, 2] += np.where(action==4, 1.0, 0.0)
        params[:, 2] += np.where(action==5, -1.0, 0.0)

        return params
    
    def take_action(self, action, num_actions=6):
        return tf.numpy_function(self._take_action, [action, num_actions], [tf.float32, tf.float32])

    def step(self, action):
        '''
        (theta, tx, ty, zx, zy)
        '''
        self.prev_params = self.params
        params = self.take_action(action)
        self.params += params
        if len(self.params_history) > 2 and tf.reduce_all(self.params_history[-2] == self.params) :
            self.revist = True
    
        # done = self._done()
        done, rewards = self._reward('not image')
        self.params_history.append(self.params)
        done = tf.cast(done, tf.float32)
        rewards = tf.cast(rewards, tf.float32)

        return self.moving, rewards, done

    def reset(self, img_fixed, img_perturb, ground_truth_params):
        self.moving = img_perturb.copy()
        self.pre_moving = img_perturb.copy()
        self.ground_truth_params = ground_truth_params
        self.params = tf.zeros(2, dtype=tf.float32)
        self.prev_params = tf.zeros(2, dtype=tf.float32)
        self.steps = 0

if __name__ == '__main__':
	train, _, _, _ = mnist_train_test(2)
	batch_generator = generate_batch(train[1000:2020])
	fixed, moving, params = next(batch_generator)
	env = environment(fixed, moving, params)
	plt.imshow(env.fixed[0, :, :, 0], cmap='gray')
	plt.show()