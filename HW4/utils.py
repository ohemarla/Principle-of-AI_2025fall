import numpy as np
import torch
from tqdm import tqdm

def compute_advantage(gamma, lmbda, td_delta):
    """
    计算广义优势估计
    """
    td_delta = td_delta.detach().numpy() # 转换为numpy数组进行计算，不传导梯度
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()    # 因为是逆序算的，最后要翻转回来
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def moving_average(a, window_size):
    """
    绘图平滑
    """
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    stats_dict = {'actor_loss': [], 'critic_loss': [], 'kl_beta': [], 'kl_d': [], 'kl_div': [], 'kl_loss': [], 'clip_loss': [], 'ratio': []}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
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
