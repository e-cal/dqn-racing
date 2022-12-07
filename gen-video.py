import argparse
import os
from collections import deque

import gym
from gym.utils.save_video import save_video

from cts_dqn import *

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="rgb_array_list")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    args = parser.parse_args()
    model_path = args.model
    model_name, _ = os.path.splitext(os.path.basename(model_path))

    agent = RacingAgent(epsilon=0)
    agent.load(model_path)

    state, _ = env.reset()
    state = process_state(state)

    total_reward = 0
    state_queue = deque([state] * 3, maxlen=3)
    t = 1
    done = False

    while t < 5000:
        state_stack = np.array(state_queue)
        action = agent.act(state_stack)
        next_state, reward, done, _, _ = env.step(action)

        total_reward += reward

        next_state = process_state(next_state)  # type: ignore
        state_queue.append(next_state)

        if done:
            print(f"length: {t}, total reward: {total_reward:.2f}")
            break

        t += 1

    save_video(env.render(), "videos", fps=env.metadata["render_fps"])
