import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tqdm
from data import *
from model import *
from env import *
from agent import *




def train():
	train, _, _, _ = mnist_train_test(2)
	batch_generator = generate_batch(train[:500])
	agent = ActorCritic(batch_generator, steps_per_game=55)
	agent.train(eps=6.5, reward_threshold=0.0, steps_per_game=55)

if __name__ == '__main__':
	train()