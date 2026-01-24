import numpy as np
import torch
from tqdm import tqdm

def compute_advantage(gamma, lmbda, td_delta):
    """
    【核心工具函数】计算广义优势估计 (Generalized Advantage Estimation, GAE)。
    
    在PPO算法中，我们需要评估每一个动作到底“有多好”。
    这个“好”是通过“优势函数(Advantage)”来衡量的。
    
    参数:
    - gamma (float): 折扣因子(Discount Factor)，范围[0,1]。
                     决定了智能体有多看重未来的奖励。
    - lmbda (float): GAE因子(Lambda)，范围[0,1]。
                     用于在"方差"和"偏差"之间做权衡。
    - td_delta (tensor): 时序差分误差 (TD Error)。
                         公式: r + gamma * V(next_state) - V(state)
                         它代表了"这一步实际发生的事情"居然比"预期"好了多少。
    
    返回:
    - advantage_list (tensor): 计算好的优势值列表，将用于后续训练 Actor。
    """
    td_delta = td_delta.detach().numpy() # 转换为 numpy 数组进行计算，不传导梯度
    advantage_list = []
    advantage = 0.0
    
    # GAE 的核心是从后往前推算 (逆序遍历)
    # 因为当前的优势不仅仅取决于现在，还取决于未来的优势
    # 公式: A_t = delta_t + (gamma * lambda) * A_{t+1}
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
        
    # 因为是逆序算的，最后要翻转回来
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def moving_average(a, window_size):
    """
    【绘图辅助函数】滑动平均。
    
    强化学习的训练曲线通常波动很大（锯齿状）。
    使用滑动平均可以将曲线变得平滑，更容易观察收敛趋势。
    
    参数:
    - a: 输入的原始奖励列表。
    - window_size: 窗口大小，越大曲线越平滑。
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    """
    【训练循环函数】适用于 On-Policy 算法 (如 PPO) 的通用训练流程。
    """
    return_list = []
    stats_dict = {'actor_loss': [], 'critic_loss': [], 'kl_beta': [], 'kl_d': [], 'kl_div': [], 'kl_loss': [], 'clip_loss': [], 'ratio': []}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {
                    'states': [], 
                    'actions': [], 
                    'next_states': [], 
                    'rewards': [], 
                    'dones': []
                }
                
                state, info = env.reset()
                done = False
                
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    
                    state = next_state
                    episode_return += reward
                
                return_list.append(episode_return)
                update_stats = agent.update(transition_dict)
                if update_stats:
                    for k, v in update_stats.items():
                        if k not in stats_dict:
                            stats_dict[k] = []
                        stats_dict[k].append(v)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, stats_dict
