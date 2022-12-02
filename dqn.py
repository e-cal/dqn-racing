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

if len(tf.config.list_physical_devices("GPU")) > 0:
    print("Using GPU")

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
def run_episode(env: gym.Env, dqn: DQN, epsilon: float):
    """Reset the environment and run through an episode using the DQN."""

    episode = []

    # initialization observations
    ob0, _ = env.reset()
    ob1 = copy.deepcopy(ob0)
    ob2 = copy.deepcopy(ob0)
    reward = 0

    step = 0
    a = np.random.randint(5)

    act_interval = 3

    mute()
    done = False
    while not done:

        # change action every few steps (prevent rapid switching)
        if step % act_interval == 0:

            # epsilon greedy action
            if np.random.random() < epsilon:
                a = np.random.randint(5)
            else:
                a, _ = dqn.get_action(join_frames(ob0, ob1, ob2))

        ob3, r, done, _, _ = env.step(a)
        step += 1
        reward += r

        # if terminal state reached
        if done:
            episode.append((join_frames(ob0, ob1, ob2), a, r, None))
            break

        # stop if reward negative [maybe move up?]
        if reward < 0:
            done = True

        episode.append((join_frames(ob0, ob1, ob2), a, r, join_frames(ob1, ob2, ob3)))
        ob0, ob1, ob2 = ob1, ob2, ob3

    unmute()
    return episode, reward


# %%
def train(
    env: gym.Env,
    dqn: DQN,
    epsilon: float,
    gamma=0.99,
    ep_decay=0.99,
    min_ep=0.01,
    iter_len=500,
    loadcp=0,
    save_interval=5,
):

    if loadcp > 0:  # load partially trained model
        fp = f"dqn-{loadcp}.hd5"
        print(f"loaded model {fp}")
        dqn.model = models.load_model(fp)  # type: ignore

    experience = []
    best_episodes = []
    best_r = [-100, -100, -100]

    for i in range(loadcp + 1, loadcp + 1000):
        print()
        print("=" * 80)
        print(f"Iteration {i}")
        print("=" * 80)

        total_steps = 0

        # save model every few training iterations
        if i % save_interval == 0:
            print(f"Saving model...")

            episode, reward = run_episode(env, dqn, epsilon=0)

            with open("training-history.txt", "a") as f:
                f.write(f"[ep {i}] length: {len(episode)}, reward: {reward}")

            fp = f"dqn-{i}.hd5"
            dqn.model.save(fp, save_format="h5")  # type: ignore
            experience += episode
            total_steps += len(episode)

        print("\nGenerating experience...")
        while total_steps < iter_len:  # gather `iter_len` steps of experience
            episode, reward = run_episode(env, dqn, epsilon)
            total_steps += len(episode)
            experience += episode
            print(
                f"[{total_steps}/500 steps] episode steps: {len(episode)} | reward: {reward}"
            )

            # keep the best 3 episodes
            if reward > min(best_r):
                best_r = best_r[1:] + [reward]
                best_episodes += episode
                if len(best_episodes) > 999 * 3:
                    best_episodes = best_episodes[-999 * 3 :]

        # keep the last 5 episodes
        if len(experience) > 999 * 5:
            experience = experience[-999 * 5 :]

        # decay epsilon
        epsilon = max(epsilon * ep_decay, min_ep)

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

        print(f"training examples: {len(examples)}")
        print(f"best reward: {best_r}")
        print(f"epsilon: {epsilon}\n")

        # get Q-table for experience
        states, actions, q_values = [], [], []
        mute()
        for s0, a, r, s1 in examples:
            states.append(np.array(s0))

            action_onehot = np.zeros(5)
            action_onehot[a] = 1
            actions.append(action_onehot)

            if s1 is None:  # terminal state
                q1 = 0
            else:
                _, q1 = dqn.get_action(s1)
            q_values.append(np.array(r + gamma * q1))
        unmute()

        # fit the model on experience and q values
        dqn.model.fit(  # type: ignore
            [np.array(states), np.array(actions)],
            np.array(q_values),
            batch_size=50,
            epochs=10,
            verbose=2,  # type: ignore
        )


# %%
env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")
dqn = DQN()
# dqn.model.summary()

# %%
train(env, dqn, epsilon=1, loadcp=0)
