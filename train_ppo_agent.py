import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import concurrent.futures
import multiprocessing as mp

# (假設原本的 gym_env, agents 導入與 preprocess_observation, compute_equity 均保持不變)
from gym_env import PokerEnv
from agents.agent import Agent
from agents.prob_agent import ProbabilityAgent
from agents.prob_agent_weak import ProbabilityAgentWeak
from agents.v10_old import V10

import json
import wandb

import logging


# 1. 在檔案全域載入勝率表
PREFLOP_TABLE = {}
if os.path.exists("preflop_equity.json"):
    with open("preflop_equity.json", "r") as f:
        PREFLOP_TABLE = json.load(f)

def compute_equity(obs, num_simulations=30):
    my_cards_raw = [c for c in obs["my_cards"] if c != -1]
    my_cards = my_cards_raw[:2] if len(my_cards_raw) >= 2 else my_cards_raw
    if len(my_cards) != 2:
        return 0.5
    community_cards = [c for c in obs["community_cards"] if c != -1]

    if len(community_cards) == 0 and len(my_cards) == 2:
        key = f"{min(my_cards)}_{max(my_cards)}"
        if key in PREFLOP_TABLE:
            return PREFLOP_TABLE[key]

    opp_discarded = list(obs.get("opp_discarded_cards", [-1, -1, -1]))
    shown_cards = set(my_cards) | set(community_cards) | {c for c in opp_discarded if c != -1}
    deck = list(range(27))
    non_shown_cards = [card for card in deck if card not in shown_cards]

    wins = 0
    for _ in range(num_simulations):
        opp_needed = 2
        board_needed = 5 - len(community_cards)
        sample_size = opp_needed + board_needed
        if sample_size > len(non_shown_cards):
            continue
        sample = random.sample(non_shown_cards, sample_size)
        opp_full = sample[:opp_needed]
        community_full = community_cards + sample[opp_needed:]

        my_hand = [PokerEnv.int_to_card(card) for card in my_cards]
        opp_hand = [PokerEnv.int_to_card(card) for card in opp_full]
        board = [PokerEnv.int_to_card(card) for card in community_full]
        evaluator = PokerEnv().evaluator
        my_rank = evaluator.evaluate(my_hand, board)
        opp_rank = evaluator.evaluate(opp_hand, board)
        if my_rank < opp_rank:
            wins += 1
    return wins / num_simulations if num_simulations > 0 else 0.0

INPUT_DIM = 16

def preprocess_observation(obs):
    street = np.array([obs["street"] / 3.0])
    my_cards = np.array([((c + 1) / 28.0) if c != -1 else 0.0 for c in obs["my_cards"]])
    if len(my_cards) < 5:
        my_cards = np.resize(my_cards, 5)
    community_cards = np.array([((c + 1) / 28.0) if c != -1 else 0.0 for c in obs["community_cards"]])
    my_bet = np.array([obs["my_bet"] / 100.0])
    opp_bet = np.array([obs["opp_bet"] / 100.0])
    min_raise = np.array([obs["min_raise"] / 100.0])
    max_raise = np.array([obs["max_raise"] / 100.0])
    equity = np.array([compute_equity(obs)])
    features = np.concatenate([street, my_cards[:5], community_cards[:5], my_bet, opp_bet, min_raise, max_raise, equity])
    return torch.tensor(features, dtype=torch.float32)


KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)

# --- 1. Actor-Critic Policy Network ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=NUM_DISCARD_CLASSES):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        state_value = self.value_head(x)
        return action_type_logits, raise_logits, discard_logits, state_value


# --- 2. PPO Agent ---
class PPOAgent:
    def __init__(self, input_dim, hidden_dim=128, lr=3e-4, gamma=0.99, eps_clip=0.05, k_epochs=4):
        self.policy = ActorCriticNetwork(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.buffer = {
            'states': [], 'valid_actions': [], 'action_types': [], 
            'raises': [], 'discards': [], 'logprobs': [], 
            'rewards': [], 'is_terminals': [], 'values': []
        }

    def store_transition(self, state, valid_actions, action_tuple, log_prob, value, reward, is_terminal):
        # 加入 detach() 確保多進程傳遞時不會帶入計算圖，節省記憶體
        self.buffer['states'].append(state.detach())
        self.buffer['valid_actions'].append(valid_actions.detach())
        self.buffer['action_types'].append(action_tuple[0])
        self.buffer['raises'].append(action_tuple[1])
        self.buffer['discards'].append(action_tuple[2])
        self.buffer['logprobs'].append(log_prob.detach())
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(is_terminal)
        self.buffer['values'].append(value.detach())

    def clear_buffer(self):
        for key in self.buffer.keys():
            self.buffer[key].clear()

    def select_action(self, state, valid_actions, min_raise, max_raise):
        with torch.no_grad():
            action_type_logits, raise_logits, discard_logits, state_value = self.policy(state)
            
            mask = (valid_actions.unsqueeze(0) == 0)
            masked_logits = action_type_logits.clone()
            masked_logits[mask] = -1e9

            action_type_dist = torch.distributions.Categorical(logits=masked_logits)
            raise_dist = torch.distributions.Categorical(logits=raise_logits)
            discard_dist = torch.distributions.Categorical(logits=discard_logits)

            action_type = action_type_dist.sample()
            raise_idx = raise_dist.sample()
            discard_idx = discard_dist.sample()

            # 只計算真正被執行的 head 的 log_prob，避免無效動作干擾梯度
            log_prob = action_type_dist.log_prob(action_type)
            if action_type.item() == PokerEnv.ActionType.RAISE.value:
                log_prob = log_prob + raise_dist.log_prob(raise_idx)
            elif action_type.item() == PokerEnv.ActionType.DISCARD.value:
                log_prob = log_prob + discard_dist.log_prob(discard_idx)

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

        action_tuple = (action_type_val, raise_amount, keep1, keep2)
        raw_action_indices = (action_type, raise_idx, discard_idx) 
        
        return action_tuple, raw_action_indices, log_prob, state_value

    def evaluate(self, states, valid_actions, action_types, raises, discards):
        action_type_logits, raise_logits, discard_logits, state_values = self.policy(states)
        
        mask = (valid_actions == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9
        
        action_type_dist = torch.distributions.Categorical(logits=masked_logits)
        raise_dist = torch.distributions.Categorical(logits=raise_logits)
        discard_dist = torch.distributions.Categorical(logits=discard_logits)

        # 只對有效動作的 head 累加 log_prob，與 select_action 保持一致
        log_probs = action_type_dist.log_prob(action_types)
        raise_mask   = (action_types == PokerEnv.ActionType.RAISE.value)
        discard_mask = (action_types == PokerEnv.ActionType.DISCARD.value)
        log_probs = log_probs + raise_mask.float()   * raise_dist.log_prob(raises)
        log_probs = log_probs + discard_mask.float() * discard_dist.log_prob(discards)

        # Entropy：action_type 恆計算；raise/discard 僅在對應動作時計入
        dist_entropy = (action_type_dist.entropy()
                        + raise_mask.float()   * raise_dist.entropy()
                        + discard_mask.float() * discard_dist.entropy())

        return log_probs, state_values.squeeze(), dist_entropy

    def update_policy(self):
        device = next(self.policy.parameters()).device # 動態取得當前神經網路所在的設備 (GPU)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer['rewards']), reversed(self.buffer['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 將所有的資料轉移到 GPU 上進行運算
        # 注意：rewards 不做 normalize，避免連輸時把「輸少」誤判為正向
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        old_states = torch.cat(self.buffer['states']).to(device)
        old_valid_actions = torch.stack(self.buffer['valid_actions']).to(device)
        old_action_types = torch.stack(self.buffer['action_types']).to(device)
        old_raises = torch.stack(self.buffer['raises']).to(device)
        old_discards = torch.stack(self.buffer['discards']).to(device)
        old_logprobs = torch.stack(self.buffer['logprobs']).to(device)
        old_values = torch.cat(self.buffer['values']).squeeze().to(device)

        advantages = rewards - old_values.detach()
        # Advantage normalization：穩定訓練，防止 advantage 尺度過大
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(
                old_states, old_valid_actions, old_action_types, old_raises, old_discards
            )

            ratios = torch.exp(logprobs - old_logprobs)
            entropy_bonus = dist_entropy.mean()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values, rewards)
            loss = actor_loss + 0.5 * critic_loss - 0.002 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping：防止 actor loss 上升時梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy_bonus.item()

        self.clear_buffer()
        return (total_actor_loss / self.k_epochs, total_critic_loss / self.k_epochs, total_entropy / self.k_epochs)


# --- 3. 多進程 Worker 函數 ---
def collect_trajectories_worker(worker_id, shared_weights, episodes_per_worker, n_sim_start=30):
    """
    這是在獨立 CPU 進程中執行的任務。
    每個 Worker 有自己獨立的環境，只使用 CPU 進行推論以收集資料。
    """
    env = PokerEnv()
    local_agent = PPOAgent(input_dim=INPUT_DIM)
    local_agent.policy.to("cpu")
    local_agent.policy.load_state_dict(shared_weights)
    
    opponent_agent = ProbabilityAgentWeak(n_sim=n_sim_start)
    # 【加入這行】強制把 V10 的日誌等級調高到 WARNING，這樣 INFO 就不會印出來了
    if hasattr(opponent_agent, 'logger'):
        opponent_agent.logger.setLevel(logging.WARNING)
        
        # 為了保險起見，把該 logger 的所有 handler 也強制設定
        for handler in opponent_agent.logger.handlers:
            handler.setLevel(logging.WARNING)
    episode_rewards = []
    win_count = 0
    loss_count = 0
    total_steps = 0

    for _ in range(episodes_per_worker):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        last_state, last_valid_acts, last_raw_acts, last_log_prob, last_val = None, None, None, None, None
        last_equity = 0.5  # 追蹤上一步的 equity，用來計算改善量
        last_street = 0
        action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}

        while not done:
            acting_agent = obs[0]["acting_agent"]
            
            if acting_agent == 0:
                # RL agent 的回合 (強制在 CPU 上運算)
                state = preprocess_observation(obs[0]).unsqueeze(0).to("cpu") 
                valid_actions_tensor = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32, device="cpu")
                current_equity = compute_equity(obs[0])

                if last_state is not None:
                    # 中間步驟不給 reward，讓訊號保持乾淨
                    local_agent.store_transition(last_state, last_valid_acts, last_raw_acts, last_log_prob, last_val, 0.0, False)

                last_equity = current_equity
                last_street = obs[0]["street"]

                action_tuple, raw_action_indices, log_prob, state_value = local_agent.select_action(
                    state, valid_actions_tensor, obs[0]["min_raise"], obs[0]["max_raise"]
                )
                
                action_counts[action_tuple[0]] += 1
                last_state = state
                last_valid_acts = valid_actions_tensor
                last_raw_acts = raw_action_indices
                last_log_prob = log_prob
                last_val = state_value
                action_to_env = action_tuple
            else:
                action_to_env = opponent_agent.act(obs[1], reward=0, terminated=False, truncated=False, info={})

            obs, reward, done, truncated, info = env.step(action_to_env)
            r = reward[0] 
            total_reward += r
            if acting_agent == 0:
                steps += 1

            if done and last_state is not None:
                # 二元勝負訊號：勝=+1, 負=-1, 平=0
                # 不用籌碼量，讓 fold 輸跟 showdown 輸受到相同懲罰
                terminal_reward = r / 100.0

                local_agent.store_transition(last_state, last_valid_acts, last_raw_acts, last_log_prob, last_val, terminal_reward, True)
                last_state = None 

        episode_rewards.append(total_reward)
        total_steps += steps
        if total_reward > 0:
            win_count += 1
        elif total_reward < 0:
            loss_count += 1

    n = len(episode_rewards)
    stats = {
        'win_rate': win_count / n if n > 0 else 0,
        'loss_rate': loss_count / n if n > 0 else 0,
        'avg_steps': total_steps / n if n > 0 else 0,
    }
    # if stats["win_rate"] > 0.4:
    #     win_rate = stats["win_rate"] * 100
    #     n_sim_start = 370/60 * (win_rate - 40) + 30
    # 回傳收集好的軌跡 Buffer 與這個 Worker 每局的總回報
    return local_agent.buffer, episode_rewards, action_counts, stats

# ==========================================
# 【新增】Eval 專用的函數與 Worker
# ==========================================
def greedy_select_action(policy_net, state, valid_actions, min_raise, max_raise):
    with torch.no_grad():
        action_type_logits, raise_logits, discard_logits, _ = policy_net(state)
        mask = (valid_actions.unsqueeze(0) == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9
        
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

def eval_worker(worker_id, shared_weights, n_sim_opp, episodes_to_play):
    env = PokerEnv()
    device = torch.device("cpu") 
    
    policy = ActorCriticNetwork(input_dim=INPUT_DIM).to(device)
    policy.load_state_dict(shared_weights)
    policy.eval() 
    
    opponent = ProbabilityAgentWeak(n_sim=n_sim_opp)
    if hasattr(opponent, 'logger'):
        opponent.logger.setLevel(logging.WARNING)
        for handler in opponent.logger.handlers:
            handler.setLevel(logging.WARNING)

    win_count, loss_count, total_reward = 0, 0, 0

    for _ in range(episodes_to_play):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            acting_agent = obs[0]["acting_agent"]
            if acting_agent == 0:
                state = preprocess_observation(obs[0]).unsqueeze(0).to(device)
                valid_acts = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32, device=device)
                action_to_env = greedy_select_action(policy, state, valid_acts, obs[0]["min_raise"], obs[0]["max_raise"])
            else:
                action_to_env = opponent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
                
            obs, reward, done, _, _ = env.step(action_to_env)
            ep_reward += reward[0]
            
        total_reward += ep_reward
        if ep_reward > 0: win_count += 1
        elif ep_reward < 0: loss_count += 1

    return win_count, loss_count, total_reward

def run_evaluation(executor, current_weights, n_sim_opp, num_workers, total_eval_episodes=500):
    episodes_per_worker = max(1, total_eval_episodes // num_workers)
    actual_episodes = episodes_per_worker * num_workers
    
    futures = []
    for i in range(num_workers):
        futures.append(executor.submit(eval_worker, i, current_weights, n_sim_opp, episodes_per_worker))
        
    total_wins, total_losses, grand_total_reward = 0, 0, 0
    for future in concurrent.futures.as_completed(futures):
        w, l, r = future.result()
        total_wins += w
        total_losses += l
        grand_total_reward += r
        
    win_rate = (total_wins / actual_episodes) * 100
    avg_reward = grand_total_reward / actual_episodes
    return win_rate, avg_reward

# --- 4. 支援多進程的主訓練迴圈 ---
def train_agent(num_episodes=10000, episodes_per_iteration=380, save_every=500, eval_every=10000, weight_path=None, weight_save_path=None):
    wandb.init(
        project="poker-rl-agent",
        name="ppo-multiprocessing-run v10fat",
        config={
            "episodes": num_episodes,
            "episodes_per_iteration": episodes_per_iteration,
            "hidden_dim": 128,
            "lr": 1e-6,
            "gamma": 0.99,
            "advantage_normalization": True,
            "grad_clip_max_norm": 0.5,
            "reward_shaping": "step_opp_minus_my_bet + terminal_clip"
        }
    )
    if weight_path is None:
        weight_path = os.path.join(os.path.dirname(__file__), "agents", "ppo_agent_weights_v10fat_bk.pth")
    if weight_save_path is None:
        weight_save_path = os.path.join(os.path.dirname(__file__), "agents", "ppo_agent_weights.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Global Agent using device: {device}")
    
    global_agent = PPOAgent(input_dim=INPUT_DIM, lr=1e-6) 
    global_agent.policy.to(device)

    # 載入預訓練 weights（如果存在）
    if os.path.exists(weight_path):
        global_agent.policy.load_state_dict(
            torch.load(weight_path, map_location=device)
        )
        print(f"Loaded pretrained weights from {weight_path}")
    else:
        print("No pretrained weights found, starting from scratch.")

    # 設定多進程數量 (留一個 CPU 核心給主進程)
    num_workers = max(1, mp.cpu_count() - 1)
    episodes_per_worker = max(1, episodes_per_iteration // num_workers)
    
    print(f"Starting training with {num_workers} parallel workers...")
    print(f"Each worker plays {episodes_per_worker} episodes per iteration.")

    episodes_played = 0

    current_n_sim = 30
    next_eval_threshold = 0  # 【新增這行】控制何時觸發期中考
    
    # 必須使用 spawn 來確保 PyTorch 多進程安全
    ctx = mp.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        while episodes_played < num_episodes:
            
            # 1. 取得最新權重並轉為 CPU 格式，以便傳遞給 Workers
            current_weights = {k: v.cpu().detach().clone() for k, v in global_agent.policy.state_dict().items()}

            if episodes_played >= next_eval_threshold:
                print(f"\n[{episodes_played}/{num_episodes}] 暫停訓練，啟動平行期中考 (500局)...")
                eval_win_rate, eval_avg_reward = run_evaluation(executor, current_weights, current_n_sim, num_workers, total_eval_episodes=500)
                
                print(f"  --> 真實勝率 (Eval Win Rate): {eval_win_rate:.1f}% | Eval Avg Reward: {eval_avg_reward:.2f}")
                
                # 根據「真實勝率」調整下一次訓練的對手強度
                if eval_win_rate > 40.0:
                    new_n_sim = int((370 / 60) * (eval_win_rate - 40) + 30)
                    current_n_sim = max(30, min(400, new_n_sim))
                    print(f"  --> 勝率達標！對手強度提升至 n_sim={current_n_sim}\n")
                else:
                    current_n_sim = 30 
                    print(f"  --> 勝率未達 40%，維持基礎難度 n_sim=30\n")
                
                wandb.log({
                    "Episode": episodes_played,
                    "Curriculum/Eval_Win_Rate": eval_win_rate,
                    "Curriculum/Eval_Avg_Reward": eval_avg_reward,
                    "Curriculum/Opponent_n_sim": current_n_sim
                })
                
                next_eval_threshold += eval_every
                
                # 如果是剛開始的第 0 局評估，測完就繼續跑底下的收集軌跡
                if episodes_played == 0:
                    pass 
            # --------------------------------------------------
            
            # 2. 發配任務給所有 Workers
            futures = []
            for i in range(num_workers):
                futures.append(
                    executor.submit(collect_trajectories_worker, i, current_weights, episodes_per_worker, current_n_sim)
                )
            
            all_episode_rewards = []
            all_action_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
            all_stats = {'win_rate': [], 'loss_rate': [], 'avg_steps': []}
            
            # 3. 等待所有 Workers 完成，並收集資料
            for future in concurrent.futures.as_completed(futures):
                worker_buffer, worker_rewards, worker_action_counts, worker_stats = future.result()
                all_episode_rewards.extend(worker_rewards)
                for k in all_action_counts:
                    all_action_counts[k] += worker_action_counts.get(k, 0)
                for k in all_stats:
                    all_stats[k].append(worker_stats[k])
                
                # 將 Worker 的資料合併到 Global Agent 的 Buffer 中
                for key in global_agent.buffer.keys():
                    global_agent.buffer[key].extend(worker_buffer[key])

            episodes_played += len(all_episode_rewards)
            
            # 4. Global Agent 進行 GPU 上的網路更新
            a_loss, c_loss, ent = global_agent.update_policy()
            
            # 5. 紀錄日誌到 WandB
            avg_reward = np.mean(all_episode_rewards)
            print(f"Episodes: {episodes_played}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Loss: {a_loss:.4f}")
            
            total_acts = sum(all_action_counts.values()) or 1
            action_names = {0:"FOLD", 1:"RAISE", 2:"CHECK", 3:"CALL", 4:"DISCARD"}
            action_str = " | ".join(f"{action_names[k]}:{all_action_counts[k]/total_acts*100:.0f}%" for k in sorted(all_action_counts))
            print(f"  Actions: {action_str}")

            avg_win_rate  = np.mean(all_stats['win_rate']) * 100
            avg_loss_rate = np.mean(all_stats['loss_rate']) * 100
            avg_steps     = np.mean(all_stats['avg_steps'])
            draw_rate     = 100 - avg_win_rate - avg_loss_rate
            print(f"  Win:{avg_win_rate:.1f}% Loss:{avg_loss_rate:.1f}% Draw:{draw_rate:.1f}% | Avg steps/ep:{avg_steps:.1f}")

            wandb.log({
                "Episode": episodes_played,
                "Reward/Batch_Average": avg_reward,
                "Loss/Actor": a_loss,
                "Loss/Critic": c_loss,
                "Loss/Entropy": ent,
                "Action/FOLD_pct":    all_action_counts[0] / total_acts * 100,
                "Action/RAISE_pct":   all_action_counts[1] / total_acts * 100,
                "Action/CHECK_pct":   all_action_counts[2] / total_acts * 100,
                "Action/CALL_pct":    all_action_counts[3] / total_acts * 100,
                "Action/DISCARD_pct": all_action_counts[4] / total_acts * 100,
                "Stats/Win_rate":   np.mean(all_stats['win_rate']) * 100,
                "Stats/Loss_rate":  np.mean(all_stats['loss_rate']) * 100,
                "Stats/Avg_steps":  np.mean(all_stats['avg_steps']),
                "Curriculum/Opponent_n_sim": current_n_sim,
            })

            # 存檔
            if episodes_played % save_every < (num_workers * episodes_per_worker):
                torch.save(global_agent.policy.state_dict(), weight_path)

    torch.save(global_agent.policy.state_dict(), weight_path)
    print("Training Completed.")

if __name__ == "__main__":
    # Windows/MacOS 需要這行來啟動多進程
    mp.set_start_method('spawn', force=True)
    # episodes_per_iteration 需要讓每個 worker 至少跑 20 局
    # 避免批次太小導致 advantage 估計不穩
    num_workers = max(1, mp.cpu_count() - 1)
    episodes_per_iteration = max(400, num_workers * 20)
    print(f"Auto-set episodes_per_iteration={episodes_per_iteration} for {num_workers} workers")
    train_agent(
        num_episodes=500000,
        episodes_per_iteration=episodes_per_iteration,
        save_every=5000,
        eval_every=10000
    )