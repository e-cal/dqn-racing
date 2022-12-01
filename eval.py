import copy
import datetime
import os
import random
import subprocess

import cv2
import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, preprocessing, regularizers
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


class ApplySoftMaxWeight(layers.Layer):
    def __init__(self, **kwargs):
        self.filter_shape = None
        super(ApplySoftMaxWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="weights", shape=[input_shape[3]])
        self.filter_shape = input_shape

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, inputs, **kwargs):
        return inputs * tf.nn.softmax(self.kernel)


class QValue:
    def __init__(self):
        self.model = None
        self.name = None
        self.checkpoint = None

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


def join_frames(o0, o1, o2):
    gray_image0 = cv2.cvtColor(cv2.resize(o0, (48, 48)), cv2.COLOR_RGB2GRAY)
    gray_image1 = cv2.cvtColor(cv2.resize(o1, (48, 48)), cv2.COLOR_RGB2GRAY)
    gray_image2 = cv2.cvtColor(cv2.resize(o2, (48, 48)), cv2.COLOR_RGB2GRAY)

    return np.array(
        [gray_image0.transpose(), gray_image1.transpose(), gray_image2.transpose()]
    ).transpose()


def get_weighted_filters(q_value, target_image, draw=True):
    o0, o1, o2 = target_image
    target_image = join_frames(o0, o1, o2)
    a, _ = q_value.get_action(target_image)
    action_onehot = np.zeros(5)
    action_onehot[a] = 1

    model = q_value.model
    pred_output = model.output  # Q Value for a specific action

    last_pool_layer = model.get_layer("pool2")
    weighted_layer = model.get_layer("weighted_filters")

    ws = weighted_layer.get_weights()[0]
    softmax_ws = np.exp(ws) / sum(np.exp(ws))

    get_vals = K.function(
        [model.input], [last_pool_layer.output[0], weighted_layer.output[0]]
    )

    pool_layer_output_val, weighted_layer_output_val = get_vals(
        [np.array([target_image]), np.array([action_onehot])]
    )

    if draw:
        fig = plt.figure(figsize=(20, 4))
        vmax = np.max(pool_layer_output_val)
        vmin = np.min(pool_layer_output_val)
        for i in range(last_pool_layer.output.shape[-1]):
            subplot = fig.add_subplot(2, 16, i + 1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.imshow(
                pool_layer_output_val[:, :, i], vmax=vmax, vmin=vmin, cmap=plt.cm.gray_r
            )

        vmax = np.max(weighted_layer_output_val)
        vmin = np.min(weighted_layer_output_val)
        for i in range(weighted_layer.output.shape[-1]):
            subplot = fig.add_subplot(2, 16, 16 + i + 1)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.imshow(
                weighted_layer_output_val[:, :, i],
                vmax=vmax,
                vmin=vmin,
                cmap=plt.cm.gray_r,
            )
            subplot.title.set_text("{:0.5f}".format(softmax_ws[i]))
    else:
        return weighted_layer_output_val


def get_heatmap(q_value, target_image):
    weighted_layer_output_val = get_weighted_filters(q_value, target_image, draw=False)
    original = target_image[-1]

    heatmap = np.zeros([96 // 8, 96 // 8])
    for i in range(weighted_layer_output_val.shape[-1]):
        heatmap += weighted_layer_output_val[:, :, i]
    h_mean = heatmap.mean()
    h_std = np.std(heatmap)
    heatmap = (heatmap - h_mean) / h_std
    heatmap = np.clip(heatmap, a_min=-1, a_max=1)
    heatmap = (heatmap + 1) / 2

    heatmap = np.uint8(255 * cv2.resize(heatmap, (96, 96)))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.35 + cv2.cvtColor(original, cv2.COLOR_RGB2BGR) * 0.65

    cv2.imwrite("/tmp/result.jpg", superimposed_img)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = preprocessing.image.load_img("/tmp/result.jpg")

    return original, heatmap, superimposed_img


def create_gif(q_value, epsilon=0):
    env = gym.make("CarRacing-v2", continuous=False, render_mode="rgb_array")
    o0, _ = env.reset()
    o1 = copy.deepcopy(o0)
    o2 = copy.deepcopy(o0)
    done = 0
    total_r = 0
    c = 0
    frames = []
    raw_frames = []

    while not done:
        if np.random.random() < epsilon:
            a = np.random.randint(5)
        else:
            a, _ = q_value.get_action(join_frames(o0, o1, o2))
        o_new, r, done, trunc, i = env.step(a)
        total_r += r
        o0, o1, o2 = o1, o2, o_new
        c += 1
        frame = env.render()

        raw_frames.append([o0, o1, o2])
        frames.append(frame)

        if c % 30 == 0:
            print("{}:{}".format(a, int(total_r)), end=", ")

    print("{}:{}".format(a, int(total_r)))
    now = datetime.datetime.now()
    imageio.mimsave(
        "car-racing-v2-{}-{:05d}-{}-{}.gif".format(
            q_value.name,
            int(q_value.checkpoint),
            int(total_r),
            now.strftime("%Y%m%d-%H%M%S"),
        ),
        frames,
        "GIF",
        **{"duration": 1.0 / 50.0}
    )
    return frames, raw_frames, total_r


def create_overlay_gif(q_value, frames, raw_frames, total_r):
    frames_overlay = []
    c = 0
    for frame, raw_frame in zip(frames, raw_frames):
        _, heatmap, _ = get_heatmap(q_value, raw_frame)
        superimposed = np.uint8(cv2.resize(heatmap, (600, 400)) * 0.35 + frame * 0.65)
        frames_overlay.append(superimposed)
        c += 1
        if c % 10 == 0:
            print(".", end="")

    print("")
    now = datetime.datetime.now()
    imageio.mimsave(
        "car-racing-v2-{}-{:05d}-{}-{}-overlay.gif".format(
            q_value.name,
            int(q_value.checkpoint),
            int(total_r),
            now.strftime("%Y%m%d-%H%M%S"),
        ),
        frames_overlay,
        "GIF",
        **{"duration": 1.0 / 50.0}
    )


def show_frames(q_value, target_images):
    num = len(target_images)
    fig = plt.figure(figsize=(11, num * 3))
    c = 1
    for i in range(len(target_images)):
        original, heatmap, superimposed = get_heatmap(q_value, target_images[i])
        subplot = fig.add_subplot(num, 4, c)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(original)
        c += 1
        gray_image = cv2.cvtColor(
            cv2.resize(original, (96 // 2, 96 // 2)), cv2.COLOR_RGB2GRAY
        )
        subplot = fig.add_subplot(num, 4, c)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(gray_image, cmap=plt.cm.gray)
        c += 1
        subplot = fig.add_subplot(num, 4, c)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(heatmap)
        c += 1
        subplot = fig.add_subplot(num, 4, c)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(superimposed)
        c += 1


def load_model(q_value, checkpoint, model="dqn0"):
    filename = "car-racing-v2-{}-{}.hd5".format(checkpoint, model)
    print("load model {}".format(filename))

    q_value.model = models.load_model(
        filename, custom_objects={"ApplySoftMaxWeight": ApplySoftMaxWeight}
    )
    q_value.name = model
    q_value.checkpoint = checkpoint


q_value = QValue()
load_model(q_value, 6, "dqn0")

frames, raw_frames, total_r = create_gif(q_value)

get_weighted_filters(q_value, raw_frames[20])

show_frames(q_value, raw_frames[50:300:20])

create_overlay_gif(q_value, frames, raw_frames, total_r)
