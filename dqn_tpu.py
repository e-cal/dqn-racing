# %%
import copy
import random

import cv2
import gym
import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# %%
MODEL_NAME = "dqn0"

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))


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

    def call(self, inputs, **kwargs):
        return inputs * tf.nn.softmax(self.kernel)


# Base model
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
q_value = DQN()
q_value.model.summary()

# %%
def join_frames(o0, o1, o2):
    gray_image0 = cv2.cvtColor(cv2.resize(o0, (48, 48)), cv2.COLOR_RGB2GRAY)
    gray_image1 = cv2.cvtColor(cv2.resize(o1, (48, 48)), cv2.COLOR_RGB2GRAY)
    gray_image2 = cv2.cvtColor(cv2.resize(o2, (48, 48)), cv2.COLOR_RGB2GRAY)

    return np.array(
        [gray_image0.transpose(), gray_image1.transpose(), gray_image2.transpose()]
    ).transpose()


# %%
def get_episode(environ, q_value, epsilon):
    episode = []
    o0, _ = environ.reset()
    o1 = copy.deepcopy(o0)
    o2 = copy.deepcopy(o0)
    total_r = 0

    if epsilon > 0:
        keep_count = 3
    else:
        keep_count = 1

    c = 0
    while True:
        if c % keep_count == 0:  # Get new action
            if np.random.random() < epsilon:
                a = np.random.randint(5)
            else:
                a, _ = q_value.get_action(join_frames(o0, o1, o2))
        c += 1
        # observation, reward, terminated, truncated, info
        o_new, r, done, trunc, inf = environ.step(a)
        total_r += r

        # Terminate episode when total reward becomes negative
        if total_r < 0:
            done = 1

        if done:
            # Terminal state is to achive more than 990 or get out of the field.
            if total_r > 990 or r < -99:
                episode.append((join_frames(o0, o1, o2), a, r, None))
            break
        else:
            episode.append((join_frames(o0, o1, o2), a, r, join_frames(o1, o2, o_new)))
        o0, o1, o2 = o1, o2, o_new

    print(
        "epsilon={}, episode length={}, total rewards={}".format(
            epsilon, len(episode), total_r
        )
    )
    return episode, total_r


# %%
def train(environ, q_value, epsilon, checkpoint=0):
    gamma = 0.99

    if checkpoint > 0:
        filename = "car-racing-v2-{}-{}.hd5".format(checkpoint, MODEL_NAME)
        print("load model {}".format(filename))
        q_value.model = models.load_model(filename)

    experience = []
    good_experience = []
    best_r = [-100, -100, -100]

    for n in range(checkpoint + 1, checkpoint + 1000):
        print("iteration {}".format(n))

        total_len = 0
        if n % 3 == 0:
            print("Testing the current performance...")
            episode, total_r = get_episode(environ, q_value, epsilon=0)
            with open("result.txt", "a") as f:
                f.write("{},{},{},{}\n".format(n, epsilon, len(episode), total_r))
            filename = "car-racing-v2-{}-{}.hd5".format(n, MODEL_NAME)
            q_value.model.save(filename, save_format="h5")
            experience += episode
            total_len += len(episode)

        while total_len < 500:
            episode, total_r = get_episode(environ, q_value, epsilon)
            total_len += len(episode)
            experience += episode

            # Keep the top 3 episodes
            if total_r > min(best_r):
                best_r = best_r[1:] + [total_r]
                good_experience += episode
                if len(good_experience) > 999 * 3:
                    good_experience = good_experience[-999 * 3 :]

        if len(experience) > 999 * 5:  # remember last 5 episodes
            experience = experience[-999 * 5 :]

        epsilon = (epsilon - 0.2) * 0.99 + 0.2

        print("Training the model...")
        # Use latest episode + past episodes (sampling) + top 3 episode (sampling)
        latest_experience = experience[-total_len:]
        past_experience = experience[:-total_len]
        examples = (
            latest_experience
            + random.sample(past_experience, min(len(past_experience), 999))
            + random.sample(good_experience, min(len(good_experience), 999))
        )

        # Show some statistics
        print("experience length={}".format(len(experience)))
        print("number of examples={}".format(len(examples)))
        print("best total reward = ", best_r)
        np.random.shuffle(examples)

        states, actions, labels = [], [], []
        for state, a, r, state_new in examples:
            states.append(np.array(state))

            action_onehot = np.zeros(5)
            action_onehot[a] = 1
            actions.append(action_onehot)

            if state_new is None:  # Terminal state
                q_new = 0
            else:
                _, q_new = q_value.get_action(state_new)
            labels.append(np.array(r + gamma * q_new))

        hist = q_value.model.fit(
            [np.array(states), np.array(actions)],
            np.array(labels),
            batch_size=50,
            epochs=10,
            verbose=0,
        )
        print("loss = {}".format(hist.history["loss"]))


# %%



# %%

checkpoint = 0
epsilon= 0.2


# This is the TPU initialization code that has to be at the beginning.


with strategy.scope():
    env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")
    q_value = DQN()
    q_value.model.summary()


z = strategy.run(train, args=(env, q_value, epsilon, checkpoint))

