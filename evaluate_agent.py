import os
import torch
import numpy as np

# 假設這些 module 都在你的專案目錄下
from gym_env import PokerEnv
from agents.prob_agent_weak import ProbabilityAgentWeak
from train_ppo_agent import ActorCriticNetwork, preprocess_observation, INPUT_DIM, KEEP_PAIRS, NUM_DISCARD_CLASSES

def greedy_select_action(policy_net, state, valid_actions, min_raise, max_raise):
    """
    評估專用的動作選擇函數：使用 argmax 選擇機率最高的動作，完全不包含隨機性。
    """
    with torch.no_grad():
        action_type_logits, raise_logits, discard_logits, _ = policy_net(state)
        
        # 遮罩掉不合法的動作
        mask = (valid_actions.unsqueeze(0) == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9
        
        # 【核心差異】：直接取 logits 最大值的 index，不使用 Categorical.sample()
        action_type = torch.argmax(masked_logits, dim=-1)
        raise_idx = torch.argmax(raise_logits, dim=-1)
        discard_idx = torch.argmax(discard_logits, dim=-1)
        
    action_type_val = action_type.item()
    raise_amount = raise_idx.item() + 1
    
    if action_type_val == PokerEnv.ActionType.RAISE.value:
        raise_amount = int(max(min(raise_amount, max_raise), min_raise))
    else:
        raise_amount = 0

    if action_type_val == PokerEnv.ActionType.DISCARD.value:
        keep1, keep2 = KEEP_PAIRS[discard_idx.item() % NUM_DISCARD_CLASSES]
    else:
        keep1, keep2 = 0, 0

    return (action_type_val, raise_amount, keep1, keep2)

def evaluate(weight_path, n_sim_opp=30, num_episodes=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化網路與對手
    policy = ActorCriticNetwork(input_dim=INPUT_DIM).to(device)
    if os.path.exists(weight_path):
        policy.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: Weights not found at {weight_path}. Evaluating untrained network.")
        
    # 【核心差異】：切換為評估模式
    policy.eval() 
    
    env = PokerEnv()
    opponent = ProbabilityAgentWeak(n_sim=n_sim_opp)
    
    # 關閉對手的 log，保持畫面乾淨
    import logging
    if hasattr(opponent, 'logger'):
        opponent.logger.setLevel(logging.WARNING)
        for handler in opponent.logger.handlers:
            handler.setLevel(logging.WARNING)
            
    win_count = 0
    loss_count = 0
    total_reward = 0
    action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    print(f"\nStarting Evaluation against n_sim={n_sim_opp} opponent for {num_episodes} episodes...")
    print("Agent is using DETERMINISTIC (Greedy) policy.")
    print("-" * 50)
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            acting_agent = obs[0]["acting_agent"]
            
            if acting_agent == 0:
                state = preprocess_observation(obs[0]).unsqueeze(0).to(device)
                valid_acts = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32, device=device)
                
                # 使用貪婪策略選擇動作
                action_tuple = greedy_select_action(
                    policy, state, valid_acts, obs[0]["min_raise"], obs[0]["max_raise"]
                )
                action_counts[action_tuple[0]] += 1
                action_to_env = action_tuple
            else:
                action_to_env = opponent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
                
            obs, reward, done, _, _ = env.step(action_to_env)
            ep_reward += reward[0]
            
        total_reward += ep_reward
        if ep_reward > 0:
            win_count += 1
        elif ep_reward < 0:
            loss_count += 1
            
        if (ep + 1) % 200 == 0:
            print(f"  Processed {ep + 1}/{num_episodes} episodes...")
            
    # 計算統計數據
    win_rate = win_count / num_episodes * 100
    loss_rate = loss_count / num_episodes * 100
    draw_rate = 100 - win_rate - loss_rate
    
    print("\n" + "=" * 50)
    print(f"Final Evaluation Results ({num_episodes} episodes):")
    print(f"Win Rate:   {win_rate:.2f}%")
    print(f"Loss Rate:  {loss_rate:.2f}%")
    print(f"Draw Rate:  {draw_rate:.2f}%")
    print(f"Avg Reward: {total_reward / num_episodes:.3f}")
    
    total_acts = sum(action_counts.values()) or 1
    action_names = {0:"FOLD", 1:"RAISE", 2:"CHECK", 3:"CALL", 4:"DISCARD"}
    action_str = " | ".join(f"{action_names[k]}:{action_counts[k]/total_acts*100:.1f}%" for k in sorted(action_counts))
    print(f"Actions:    {action_str}")
    print("=" * 50)

if __name__ == "__main__":
    weight_file = os.path.join(os.path.dirname(__file__), "agents", "ppo_agent_weights.pth")
    # 跑 1000 局來獲得比較穩定且具代表性的勝率
    evaluate(weight_path=weight_file, n_sim_opp=30, num_episodes=1000)