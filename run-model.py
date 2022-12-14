import argparse
from collections import deque

import gym

from mod_disc_dqn import *

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode="human")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    args = parser.parse_args()
    model_path = args.model

    agent = RacingAgent(epsilon=0)
    agent.load(model_path)

    for ep in range(3):
        state, _ = env.reset()
        state = process_state(state)

        total_reward = 0
        state_queue = deque([state] * 3, maxlen=3)
        t = 1
        done = False

        while True:
            state_stack = np.array(state_queue)
            action = agent.act(state_stack)
            next_state, reward, done, _, _ = env.step(action)

            total_reward += reward

            next_state = process_state(next_state)  # type: ignore
            state_queue.append(next_state)

            if done:
                print(
                    f"episode: {ep}/{3}, length: {t}, total reward: {total_reward:.2f}"
                )
                break

            t += 1
