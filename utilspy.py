import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tqdm
import cv2
import glob

from data import *
from model import *
from env import *
from agent import *

def rotation_translation_warp(image, params):
    temp = tf.keras.preprocessing.image.apply_affine_transform(
                                image[0], 
                                theta=0, 
                                tx=params[0][1], 
                                ty=params[0][2], shear=0, 
                                zx=1, 
                                zy=1, row_axis=0, col_axis=1,
                                channel_axis=2, fill_mode='constant', cval=0.0, order=1)
    temp = tf.keras.preprocessing.image.apply_affine_transform(
                                temp, 
                                theta=params[0][0]*30, 
                                tx=0, 
                                ty=0, shear=0, 
                                zx=1, 
                                zy=1, row_axis=0, col_axis=1,
                                channel_axis=2, fill_mode='constant', cval=0.0, order=1)
    return temp

def save_to_video(images_path, output_name):
    img_array = []
    # for filename in glob.glob('/content/drive/MyDrive/Reinforcemen Learning/images/*.jpg'):
    for filename in glob.glob(images_path+'/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter(output_name+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def test(model, test_generator, steps_per_game=30, save_images_path=None):
    # test_fixed, test_moving, test_params = next(test_generator)
    test_fixed, test_moving, test_params = test_generator
    test_env = environment(test_fixed, test_moving, test_params, steps_per_game=100)
    done = False

    state = test_env.moving
    initial_state_shape = state.shape
    i = 0
        # while not done:
    for t in tf.range(steps_per_game):
#             action_prob, value, state_h, state_c = self.model([state, fixed, state_h, state_c])
        action_logits_t, value = model([state, test_fixed])
#             print(action_logits_t)
        action = tf.random.categorical(action_logits_t, 1)[0, 0]
#         action_probs_t = tf.nn.softmax(action_logits_t)
#             print(action)
#         values = values.write(t, tf.squeeze(value))
#         action_probs = action_probs.write(t, action_probs_t[0, action])

        state, reward, done = test_env.step(action)
        state.set_shape(initial_state_shape)
#         rewards = rewards.write(t, tf.squeeze(reward))
        fixed = tf.cast(test_fixed[0]*255, tf.uint8)
        # save the images 
        if save_images_path is not None:
            temp = rotation_translation_warp(test_moving, test_env.params)
            diff = tf.cast((test_env.fixed[0]-temp)**2 * 255, tf.uint8)
            current_state = tf.cast(state[0]*255, tf.uint8)
            concat = tf.concat([fixed, current_state, diff], axis=1)
            current_state = tf.io.encode_jpeg(current_state, quality=100, format='grayscale')
            diff = tf.io.encode_jpeg(diff, quality=100, format='grayscale')
            concat = tf.io.encode_jpeg(concat, quality=100, format='grayscale')
            tf.io.write_file('images/current state/'+str(i)+'.jpg', current_state)
            tf.io.write_file('images/current diff/'+str(i)+'.jpg', diff)
            tf.io.write_file('images/current concatenated/'+str(i)+'.jpg', concat)

        i += 1
        # if tf.cast(done, tf.bool):
        #     break
        if tf.cast(tf.math.reduce_euclidean_norm(test_env.moving-test_env.fixed, axis=[1,2,3]), tf.float32) < 3.50:
            break

    print('solve at step ', i)

    #----------------------------transfer to video---------------------------#
    if save_images_path is not None:
        save_to_video(save_images_path+'images/current state', 'state changes')
        save_to_video(save_images_path+'images/current diff', 'difference changes')
        save_to_video(save_images_path+'images/current concatenated', 'concat')
    #------------------------------------------------------------------------#

    temp = rotation_translation_warp(test_moving, -test_params)
    gt = compare(test_env.fixed, test_env.original_moving)[:,:,0]
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
    ax1.imshow(test_env.fixed[0, :, :, 0], cmap='gray')
    ax1.set_title('fixed')
    ax2.imshow(test_env.moving[0, :, :, 0], cmap='gray')
    ax2.set_title('moved')
    ax3.imshow(test_env.original_moving[0, :, :, 0], cmap='gray')
    ax3.set_title('initial')
    ax4.imshow((test_env.fixed[0,:,:,0]-test_env.moving[0,:,:,0])**2, cmap='gray')
    ax4.set_title('diff')
    ax5.imshow((gt-test_env.fixed[0,:,:,0])**2, cmap='gray')
    ax5.set_title('dft diff')
    ax6.imshow((test_env.fixed[0,:,:,0]-temp[:,:,0])**2, cmap='gray')
    ax6.set_title('gt diff')
    plt.show()

    # print('ground truth: ', test_env.ground_truth_params)
    # print('predicted params: ', -test_env.params)
    tf.print('image distance: ', tf.math.reduce_euclidean_norm(test_env.moving[0,:,:,0]-test_env.fixed[0,:,:,0]))
    tf.print('dft distance: ', tf.math.reduce_euclidean_norm(gt-test_env.fixed[0,:,:,0]))
    tf.print('gt image distance: ', tf.math.reduce_euclidean_norm(temp[:,:,0]-test_env.fixed[0,:,:,0]))

if __name__ == '__main__':
	train, _, _, _ = mnist_train_test(2)
	test_batch_generator = generate_batch(train[-500:])
	for _ in range(5):
    	test_fixed, test_moving, test_params = next(test_batch_generator)
    	test(model, (test_fixed, test_moving, test_params), steps_per_game=35)#, save_images_path='/content/drive/MyDrive/Reinforcemen Learning/')