import tensorflow as tf
# print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import tqdm
from data import *
from model import *
from env import *


class ActorCritic:
    def __init__(self, data_generator, eps=2.0, gamma=0.9, steps_per_game=25, modelpath=None):
        self.steps_per_game = steps_per_game
        self.data_generator = data_generator
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = gamma
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.eps = eps
        fixed, moving, params = next(data_generator)
        self.env = environment(fixed, moving, params, eps=self.eps)
        if modelpath is None:
            self.model = MyModel()
        else:
            self.model = tf.keras.models.load_model(modelpath)
        self.episodes = 0
        self.history_reward = []

    # def run_episode(self, img_fixed, img_perturb, ground_truth_params, steps_per_game=20):
    def run_episode(self):
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        fixed, moving, params = next(self.data_generator)
        self.env = environment(fixed, moving, params, self.steps_per_game, eps=self.eps)
        # env = environment(img_fixed, img_perturb, ground_truth_params, steps_per_game=steps_per_game)
        done = False
#         state_h = tf.random.uniform(shape=(1, 256), maxval=1.0, dtype=tf.float32)
#         state_c = tf.random.uniform(shape=(1, 256), maxval=1.0, dtype=tf.float32)
        state = self.env.moving
        initial_state_shape = state.shape

        # while not done:
        for t in tf.range(self.steps_per_game):
#             action_prob, value, state_h, state_c = self.model([state, fixed, state_h, state_c])
            action_logits_t, value = self.model([state, fixed])
#             print(action_logits_t)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
#             print(action)
            values = values.write(t, tf.squeeze(value))
            action_probs = action_probs.write(t, action_probs_t[0, action])

            state, reward, done = self.env.step(action)
            state.set_shape(initial_state_shape)

            if tf.math.reduce_euclidean_norm(self.env.params-self.env.ground_truth_params, axis=[1])<1e-3:
                reward = tf.constant(50, dtype=tf.float32)
                done = True
            rewards = rewards.write(t, tf.squeeze(reward))
            
            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards
    
    def get_expected_return(self, rewards: tf.Tensor,
                            standardize=True):
        """Compute expected returns per timestep."""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                    (tf.math.reduce_std(returns) + self.eps))
        return returns

    def compute_loss(self,
            action_probs: tf.Tensor,  
            values: tf.Tensor,  
            returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    def train_step(self):
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode() 

            # Calculate expected returns
            returns = self.get_expected_return(rewards)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

            # Calculating loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward
    
    def train(self, eps=2.0, reward_threshold=5.0, steps_per_game=25, fine_tune=False):
#         %%time
        self.steps_per_game = steps_per_game
        self.eps = eps
        max_episodes = 1000000

        running_reward = 0

        # Discount factor for future rewards
        gamma = 0.99

        with tqdm.trange(max_episodes) as t:
            for i in t:
                episode_reward = self.train_step()

                running_reward = episode_reward*0.01 + running_reward*.99

                t.set_description(f'Episode {self.episodes}')
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)
                # t.set_postfix(episode_reward=tf.keras.backend.get_value(episode_reward),
                #               running_reward=tf.keras.backend.get_value(running_reward))

                if not fine_tune:
                    save_steps = 5000
                else:
                    save_steps = 1000

                self.history_reward.append(running_reward)
                if self.episodes > 500 and running_reward > reward_threshold:
                    break
                    
                if self.episodes % save_steps == 0:
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                    ax1.imshow(self.env.fixed[0, :, :, 0], cmap='gray')
                    ax1.set_title('fixed')
                    ax2.imshow(self.env.moving[0, :, :, 0], cmap='gray')
                    ax2.set_title('moved')
                    ax3.imshow(self.env.original_moving[0, :, :, 0], cmap='gray')
                    ax3.set_title('original moving')
                    ax4.imshow((self.env.fixed[0,:,:,0]-self.env.moving[0,:,:,0])**2, cmap='gray')
                    ax4.set_title('difference')
                    plt.show()

                    print()
                    print('ground truth: ', self.env.ground_truth_params)
                    print('predicted params: ', -self.env.params)
                    print('image distance: ', tf.math.reduce_euclidean_norm(self.env.moving-self.env.fixed, axis=[1,2,3]))
                    print(f'Episode {i}: average reward: {running_reward}')
                    path = 'translation and rotation/model'+str(self.episodes)+'/'
                    self.model.save(path)

                    if self.episodes > 500 and running_reward > reward_threshold:
                        break

                self.episodes += 1


        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
        path = 'translation and rotation/model'+str(self.episodes)+'/'
        self.model.save(path)
        plt.plot(self.history_reward)
        plt.show()

# if __name__ == '__main__':
#     train, _, _, _ = mnist_train_test(2)
#     batch_generator = generate_batch(train[1000:2020])
#     agent = ActorCritic(batch_generator, steps_per_game=55)
#     agent.train(eps=4.0, reward_threshold=0.0)
