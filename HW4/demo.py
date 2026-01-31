import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

def run_demo():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    model_path = 'HW4/Model/ppo_cartpole_model.pth'
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    num_episodes = 5    # 模拟5局游戏
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            state_tensor = torch.tensor([state], dtype=torch.float).to(device)
            with torch.no_grad():
                probs = actor(state_tensor)
            action = torch.argmax(probs).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(0.02)
        print(f"第 {i+1} 局得分: {total_reward}")
        time.sleep(1)
    env.close()

if __name__ == "__main__":
    run_demo()
