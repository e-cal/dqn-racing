import argparse
import os
from collections import deque

import gym
from gym.utils.save_video import save_video

from score_decay import *

NVIDS = 3

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="rgb_array_list")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    args = parser.parse_args()
    model_path = args.model
    model_name, _ = os.path.splitext(os.path.basename(model_path))

    agent = RacingAgent(epsilon=0)
    agent.load(model_path)

    for i in range(NVIDS):
        state, _ = env.reset()
        state = process_state(state)

        total_reward = 0
        n_rewards = 0
        state_queue = deque([state] * 3, maxlen=3)
        t = 1

        while True:
            state_stack = np.array(state_queue)
            action = agent.act(state_stack)
            next_state, reward, done, _, _ = env.step(action)

            n_rewards = n_rewards + 1 if t > 1000 and reward < 0 else 0

            if n_rewards >= 100:
                reward *= 10

            total_reward += reward

            next_state = process_state(next_state)  # type: ignore
            state_queue.append(next_state)

            if done or total_reward < 0:
                print(f"length: {t}, total reward: {total_reward:.2f}")
                break

            t += 1

        save_video(
            env.render(),  # type: ignore
            video_folder="videos",
            name_prefix=model_name,
            episode_index=i,
            episode_trigger=lambda _: True,
            fps=env.metadata["render_fps"],
        )

    # os.rename("videos/rl-video-episode-0.mp4", f"videos/{model_name}.mp4")
