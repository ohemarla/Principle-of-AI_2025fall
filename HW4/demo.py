import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

# 1. 定义与训练时一致的网络结构
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

def run_demo():
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 环境配置
    env_name = 'CartPole-v1'
    # render_mode='human' 表示会弹窗显示动画
    env = gym.make(env_name, render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128  # 必须与训练时保持一致

    # 初始化网络
    actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
    
    # 加载模型参数
    model_path = 'HW4/Model/ppo_cartpole_model.pth'

    # 加载权重
    # map_location确保在没有GPU的机器上也能用CPU运行
    state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval() # 切换到评估模式
    print("模型加载成功！")

    # 开始演示
    num_episodes = 5
    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        print(f"开始演示第 {i+1}/{num_episodes} 局...")
        
        while not done:
            env.render()
            
            # 模型推理
            state_tensor = torch.tensor([state], dtype=torch.float).to(device)
            with torch.no_grad():
                probs = actor(state_tensor)
            
            # 选择概率最大的动作 (确定性策略演示效果通常更好)
            action = torch.argmax(probs).item()
            # 或者也可以继续采样:
            # action_dist = torch.distributions.Categorical(probs)
            # action = action_dist.sample().item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # 稍微限制一下帧率，防止动画太快太鬼畜
            time.sleep(0.02)
            
        print(f"第 {i+1} 局得分: {total_reward}")
        time.sleep(1) # 局间休息

    env.close()
    print("演示结束。")

if __name__ == "__main__":
    run_demo()
