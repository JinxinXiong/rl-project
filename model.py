import tensorflow as tf
# print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tqdm
from data import *

class MyModel(Model):
    def __init__(self, num_actions=6):
        super(MyModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(8, kernel_size=3, strides=(1, 1), padding='same', activation='elu')
        self.mp1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding='same', activation='elu')
        self.mp2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', activation='elu')
        self.mp3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='elu')
        self.actor = tf.keras.layers.Dense(num_actions, name='actor')
        self.critic = tf.keras.layers.Dense(1, name='critic')
        
    @tf.function
    def call(self, inputs):
        '''
        moving, fixed: image batch with shape (batch_size, height, width, channels)
        return actor, critic, hidden and cell state with shape (1, 256)
        '''
        moving, fixed = inputs
        x = tf.keras.layers.concatenate([moving, fixed], axis=-1)
        x = tf.cast(x, dtype=tf.float32)
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.fc1(x)

        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

def main():
	train, _, _, _ = mnist_train_test(2)
	batch_generator = generate_batch(train[1000:2020])
	fixed, moving, params = next(batch_generator)
	model = MyModel()
	a, critic = model([moving, fixed])
	print(a.shape, critic.shape)


if __name__ == '__main__':
	main()


