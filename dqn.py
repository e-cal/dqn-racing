# %%
import copy
import os
import random
import sys

import cv2
import gym
import numpy as np
import tensorflow as tf
from keras import layers, models

stdout = sys.stdout


def mute():
    sys.stdout = open(os.devnull, "w")


def unmute():
    sys.stdout = stdout


# %%
class SoftMax(layers.Layer):
    def __init__(self, **kwargs):
        self.filter_shape = None
        super(SoftMax, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="weights", shape=[input_shape[3]])
        self.filter_shape = input_shape

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs):
        return inputs * tf.nn.softmax(self.kernel)


class DQN:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        cnn_input = layers.Input(shape=(48, 48, 3), name="cnn_input")
        cnn1 = layers.Conv2D(
            16, (5, 5), padding="same", use_bias=True, activation="relu", name="cnn1"
        )(cnn_input)
        pool1 = layers.MaxPooling2D((2, 2), name="pool1")(cnn1)
        cnn2 = layers.Conv2D(
            16, (5, 5), padding="same", use_bias=True, activation="relu", name="cnn2"
        )(pool1)
        pool2 = layers.MaxPooling2D((2, 2), name="pool2")(cnn2)
        weighted_filters = SoftMax(name="weighted_filters")(pool2)

        cnn_flatten = layers.Flatten(name="flatten")(weighted_filters)
        action_input = layers.Input(shape=(5,), name="action_input")
        combined = layers.concatenate([cnn_flatten, action_input], name="concat")
        hidden1 = layers.Dense(2048, activation="relu", name="dense1")(combined)
        hidden2 = layers.Dense(1024, activation="relu", name="dense2")(hidden1)
        hidden3 = layers.Dense(512, activation="relu", name="dense3")(hidden2)
        q_value = layers.Dense(1, name="output")(hidden3)

        model = models.Model(inputs=[cnn_input, action_input], outputs=q_value)
        model.compile(loss="mse")
        return model

    def get_action(self, state):
        states = []
        actions = []
        for a in range(5):
            states.append(np.array(state))
            action_onehot = np.zeros(5)
            action_onehot[a] = 1
            actions.append(action_onehot)

        q_values = self.model.predict([np.array(states), np.array(actions)])
        optimal_action = np.argmax(q_values)
        return optimal_action, q_values[optimal_action][0]


# %%
dqn = DQN()
dqn.model.summary()

# %%
def join_frames(o0, o1, o2):
    gray_image0 = cv2.cvtColor(cv2.resize(o0, (48, 48)), cv2.COLOR_RGB2GRAY)  # type: ignore
    gray_image1 = cv2.cvtColor(cv2.resize(o1, (48, 48)), cv2.COLOR_RGB2GRAY)  # type: ignore
    gray_image2 = cv2.cvtColor(cv2.resize(o2, (48, 48)), cv2.COLOR_RGB2GRAY)  # type: ignore

    return np.array(
        [gray_image0.transpose(), gray_image1.transpose(), gray_image2.transpose()]
    ).transpose()


# %%
def run_episode(env: gym.Env, dqn: DQN, epsilon):
    episode = []
    o0, _ = env.reset()
    o1 = copy.deepcopy(o0)
    o2 = copy.deepcopy(o0)
    reward = 0

    act_interval = 3

    step = 0
    a = np.random.randint(5)

    mute()

    done = False
    while not done:
        if step % act_interval == 0:
            if np.random.random() < epsilon:
                a = np.random.randint(5)
            else:
                a, _ = dqn.get_action(join_frames(o0, o1, o2))

        step += 1

        o3, r, done, _, _ = env.step(a)

        reward += r

        # stop if reward negative
        if reward < 0:
            done = True

        if done and (reward > 990 or r < -99):  # if terminal state reached
            episode.append((join_frames(o0, o1, o2), a, r, None))
            break

        episode.append((join_frames(o0, o1, o2), a, r, join_frames(o1, o2, o3)))
        o0, o1, o2 = o1, o2, o3

    unmute()
    return episode, reward


# %%
def train(env: gym.Env, dqn: DQN, epsilon: float, gamma=0.99, checkpoint=0):
    if checkpoint > 0:
        filename = f"dqn-{checkpoint}.hd5"
        print(f"loaded model {filename}")
        dqn.model = models.load_model(filename)  # type: ignore

    experience = []
    best_episodes = []
    best_r = [-100, -100, -100]

    for ep in range(checkpoint + 1, checkpoint + 1000):
        print()
        print("=" * 80)
        print(f"Iteration {ep}")
        print("=" * 80)

        total_steps = 0
        if ep % 3 == 0:
            print("=" * 80)
            print(f"Saving model...")
            episode, reward = run_episode(env, dqn, epsilon=0)
            with open("result.txt", "a") as f:
                f.write(f"[ep {ep}] length: {len(episode)}, reward: {reward}")
            filename = f"dqn-{ep}.hd5"
            dqn.model.save(filename, save_format="h5")  # type: ignore
            experience += episode
            total_steps += len(episode)

        print("\nGenerating experience...")
        while total_steps < 500:
            episode, reward = run_episode(env, dqn, epsilon)
            print(
                f"[{total_steps}/500 steps] episode steps: {len(episode)} | reward: {reward}"
            )
            total_steps += len(episode)
            experience += episode

            # keep the top 3 episodes
            if reward > min(best_r):
                best_r = best_r[1:] + [reward]
                best_episodes += episode
                if len(best_episodes) > 999 * 3:
                    best_episodes = best_episodes[-999 * 3 :]

        if len(experience) > 999 * 5:  # remember last 5 episodes
            experience = experience[-999 * 5 :]

        epsilon = (epsilon - 0.2) * 0.99 + 0.2

        print("\nFitting model...")

        # Use latest episode + sample of past episodes + sample of top best episodes
        latest_experience = experience[-total_steps:]
        past_experience = experience[:-total_steps]
        examples = (
            latest_experience
            + random.sample(past_experience, min(len(past_experience), 999))
            + random.sample(best_episodes, min(len(best_episodes), 999))
        )
        np.random.shuffle(examples)

        # Show some statistics
        print(f"training examples: {len(examples)}")
        print(f"best reward: {best_r}\n")

        mute()
        states, actions, labels = [], [], []
        for state, a, r, state_new in examples:
            states.append(np.array(state))

            action_onehot = np.zeros(5)
            action_onehot[a] = 1
            actions.append(action_onehot)

            if state_new is None:  # Terminal state
                q_new = 0
            else:
                _, q_new = dqn.get_action(state_new)
            labels.append(np.array(r + gamma * q_new))
        unmute()

        hist = dqn.model.fit(  # type: ignore
            [np.array(states), np.array(actions)],
            np.array(labels),
            batch_size=50,
            epochs=10,
            verbose=2,  # type: ignore
        )


# %%
env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")
dqn = DQN()
# dqn.model.summary()

# %%
train(env, dqn, epsilon=0.2, checkpoint=0)
