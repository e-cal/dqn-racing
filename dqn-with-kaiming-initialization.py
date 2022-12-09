import os
import random
from collections import deque, namedtuple

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NAME = "vanilla-kaiming-initialization"
SAVE_FREQ = 25

def process_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)  # type: ignore
    state = state.astype(float)
    state /= 255.0
    return state

class DQN(nn.Module):
    def __init__(
        self,
        ninputs,
        noutputs,
        seed=None,
        initialization="random"
    ):
        super(DQN, self).__init__()

        if seed:
            self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(
            in_channels=ninputs,
            out_channels=6,
            kernel_size=(7, 7),
            stride=3,
        )

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=(4, 4),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(432, 216)
        self.fc2 = nn.Linear(216, noutputs)

        if initialization == "kaiming":
            nn.init.kaiming_normal_(self.conv1.weight)
            nn.init.kaiming_normal_(self.conv2.weight)
            nn.init.kaiming_normal_(self.fc1.weight)
            nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            pred = self.forward(state)

        return pred

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = (
            torch.from_numpy(np.array([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.array([e.action for e in experiences if e is not None]))
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.array([e.reward for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.array([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(np.array([e.done for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class RacingAgent:
    def __init__(
        self,
        actions=[
            # (steer [-1,1], gas [0,1], break [0,1])
            (-1, 1, 0.2),
            (0, 1, 0.2),
            (1, 1, 0.2),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            (-1, 0, 0.2),
            (0, 0, 0.2),
            (1, 0, 0.2),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ],
        gamma=0.95,  # discount rate
        epsilon=1.0,  # random action rate
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001,
        tau=1e-3,  # soft update discount
        update_main_network_freq=1,
        hard_update=False,
        dqn_loss="mse",
        act_interval=2,
        buffer_size=5000,
        batch_size=64,
        save_freq=25,
        seed=None,
        initialization="random"
    ):
        self.obs_shape = (96, 96, 3)
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.update_main_network_freq = update_main_network_freq
        self.hard_update = hard_update
        self.dqn_loss = dqn_loss
        self.initialization = initialization

        self.seed = seed if seed is not None else np.random.randint(1000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.dqn_behavior = DQN(3, len(self.actions), self.seed, self.initialization).to(device)
        self.dqn_target = DQN(3, len(self.actions), self.seed, self.initialization).to(device)
        self.optimizer = optim.Adam(self.dqn_behavior.parameters(), lr=learning_rate)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(
            action_size=len(self.actions),
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            seed=seed,
        )
        self.act_interval = act_interval
        self.save_freq = save_freq
        self.training_steps = 0
        

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() > self.epsilon:
            action_values = self.dqn_behavior.predict(state)
            aind = np.argmax(action_values.cpu().data.numpy())

        else:
            aind = random.randrange(len(self.actions))

        return self.actions[aind]

    def soft_update_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(
            self.dqn_target.parameters(), self.dqn_behavior.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update_target(self):
        """Hard update model parameters."""
        for target_param, local_param in zip(
            self.dqn_target.parameters(), self.dqn_behavior.parameters()
        ):
            target_param.data.copy_(
                local_param.data
            )        

    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
    
        # get Q tables from both networks
        q_targets_next = self.dqn_target(next_states).detach().max(1)[0]
        q_targets = (rewards + self.gamma * q_targets_next * (1 - dones)).unsqueeze(1)

        q_preds = self.dqn_behavior(states)
        q_preds = q_preds.gather(1, actions.unsqueeze(1))
        
        # fit behavior dqn
        self.dqn_behavior.train()
        self.optimizer.zero_grad()
        if self.dqn_loss == "mse":
            loss = F.mse_loss(q_preds, q_targets)
        elif self.dqn_loss == "huber":
            loss = F.huber_loss(q_preds, q_targets)
        loss.backward()
        self.training_steps += 1
        
        # Frequency at which main network weights should be updated
        if (self.training_steps % self.update_main_network_freq == 0):
            self.optimizer.step()
        
        if not self.hard_update:
            self.soft_update_target()
        else:
            self.hard_update_target()
        
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, fp):
        checkpoint = torch.load(fp)
        self.dqn_behavior.load_state_dict(checkpoint["model_state"])
        self.dqn_target.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, epoch, steps, reward, epsilon, loss):
        print(f"saving model to models/{NAME}-{epoch}.pth")

        fp = f"{NAME}-hist.csv"

        if not os.path.exists(fp):
            with open(fp, "w") as f:
                f.write(f"epoch,epsilon,steps,reward,loss\n")

        with open(fp, "a") as f:
            f.write(f"{epoch},{epsilon},{steps},{reward},{loss}\n")

        torch.save(
            {
                "model_state": self.dqn_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"models/{NAME}-{epoch}.pth",
        )

    def train(self, env: gym.Env, start_ep: int, end_ep: int, max_neg=25):
        print(f"Starting training from ep {start_ep} to {end_ep}...")
        for ep in range(start_ep, end_ep + 1):
            state, _ = env.reset()
            state = process_state(state)

            total_reward = 0
            n_rewards = 0
            state_queue = deque([state] * 3, maxlen=3)  # queue 3 states
            t = 1
            done = False

            while True:
                state_stack = np.array(state_queue)
                action = self.act(state_stack)

                reward = 0
                for _ in range(self.act_interval + 1):
                    next_state, r, done, _, _ = env.step(action)
                    reward += r
                    if done:
                        break

                # end episode if continually getting negative reward
                n_rewards = n_rewards + 1 if t > 100 and reward < 0 else 0

                total_reward += reward

                next_state = process_state(next_state)  # type: ignore
                state_queue.append(next_state)
                next_state_stack = np.array(state_queue)

                self.memory.add(
                    state_stack,
                    self.actions.index(action),
                    reward,
                    next_state_stack,
                    done,
                )

                if done or n_rewards >= max_neg or total_reward < 0:
                    print(
                        f"episode: {ep}/{end_ep}, length: {t}, total reward: {total_reward:.2f}, epsilon: {self.epsilon:.2f}"
                    )
                    break

                if len(self.memory) > self.batch_size:
                    loss = self.learn()

                t += 1

            if ep % self.save_freq == 0:
                self.save(ep, t, total_reward, epsilon, loss)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Path to partially trained model (hd5)",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=1,
        help="starting episode (to continue training from)",
    )
    parser.add_argument("-x", "--end", type=int, default=1000, help="ending episode")
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1.0,
        help="Starting epsilon (default: 1)",
    )

    args = parser.parse_args()

    if args.model:
        print("loading a model, make sure start and epsilon are set correctly")

    return args.model, args.start, args.end, args.epsilon


if __name__ == "__main__":
    model_path, start, end, epsilon = get_args()

    env = gym.make("CarRacing-v2")
    agent = RacingAgent(
        actions=[
            # (steer [-1,1], gas [0,1], break [0,1])
            (-1, 1, 0.2),
            (0, 1, 0.2),
            (1, 1, 0.2),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            (-1, 0, 0.2),
            (0, 0, 0.2),
            (1, 0, 0.2),
            (-1, 0, 0),
            (0, 0, 0),
            (1, 0, 0),
        ],
        gamma=0.95,  # discount rate
        epsilon=epsilon,  # random action rate
        epsilon_min=0.1,
        epsilon_decay=0.9999,
        learning_rate=0.001,
        tau=1e-3,  # soft update discount
        update_main_network_freq=1,
        hard_update=False,
        dqn_loss="huber",
        act_interval=2,
        buffer_size=5000,
        batch_size=64,
        save_freq=SAVE_FREQ,
        seed=420,
        initialization="kaiming"
    )

    if model_path:
        agent.load(model_path)

    agent.train(env, start, end)

    env.close()
