import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import itertools

# Import our poker environment and opponent agent classes.
from gym_env import PokerEnv
from agents.agent import Agent
from agents.prob_agent import ProbabilityAgent
from agents.test_agents import FoldAgent, CallingStationAgent, AllInAgent, RandomAgent

# ==========================================
# 🚀 挂载我们的 O(1) 核武器级胜率字典
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
print("📥 正在挂载绝对牌力字典与 Pre-flop 胜率表...")
try:
    with open(os.path.join(current_dir, "lookup_table_7cards.pkl"), "rb") as f:
        GLOBAL_LOOKUP_TABLE = pickle.load(f)
    with open(os.path.join(current_dir, "preflop_table.pkl"), "rb") as f:
        GLOBAL_PREFLOP_TABLE = pickle.load(f)
    print("✅ 字典挂载完毕！RL 训练机已挂载极速引擎！")
except Exception as e:
    print(f"❌ 字典读取失败: {e}。请确保 .pkl 文件放在 train_rl_agent.py 同级目录下！")
    # 如果没找到字典，程序不能盲目往下跑，直接停止
    exit()

# --- Helper Functions for Preprocessing and Equity Calculation ---

def compute_equity(obs):
    """
    100% 精确的真实胜率计算引擎！彻底替代原本缓慢且充满误差的蒙特卡洛。
    为神经网络提供绝对准确的视角。
    """
    street = obs.get("street", 0)
    my_cards_raw = [c for c in obs["my_cards"] if c != -1]
    community_cards = [c for c in obs["community_cards"] if c != -1]
    opp_discarded = [c for c in obs.get("opp_discarded_cards", [-1, -1, -1]) if c != -1]

    # 1. 翻牌前 (Pre-flop)：瞬间 O(1) 查表
    if street == 0 and len(my_cards_raw) == 5:
        return GLOBAL_PREFLOP_TABLE.get(tuple(sorted(my_cards_raw)), 0.5)

    # 2. 翻牌后 (Post-flop)：保留 2 张牌计算
    my_cards = my_cards_raw[:2] if len(my_cards_raw) >= 2 else my_cards_raw
    if len(my_cards) != 2 or GLOBAL_LOOKUP_TABLE is None:
        return 0.5

    # 3. 穷举计算真实胜率 (吃透对手弃牌情报)
    known_cards = set(my_cards) | set(community_cards) | set(opp_discarded)
    unknown_cards = [c for c in range(27) if c not in known_cards]
    
    wins = ties = total = 0
    board_needed = 5 - len(community_cards)

    # 如果是 River 阶段 (board_needed=0)，itertools.combinations([], 0) 会正常处理
    for opp_2 in itertools.combinations(unknown_cards, 2):
        rem_deck = [c for c in unknown_cards if c not in opp_2]
        for future_comm in itertools.combinations(rem_deck, board_needed):
            my_7 = tuple(sorted(my_cards + community_cards + list(future_comm)))
            opp_7 = tuple(sorted(list(opp_2) + community_cards + list(future_comm)))
            
            my_score = GLOBAL_LOOKUP_TABLE[my_7]
            opp_score = GLOBAL_LOOKUP_TABLE[opp_7]
            
            if my_score < opp_score: wins += 1
            elif my_score == opp_score: ties += 1
            total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5

# New variant: 5 hole card slots, 27-card deck. Feature dim = 1 + 5 + 5 + 1+1+1+1+1 = 16
INPUT_DIM = 16

def preprocess_observation(obs):
    """
    Converts the observation into a feature tensor for the new variant.
    Features: street(1), my_cards(5), community_cards(5), my_bet, opp_bet, min_raise, max_raise, equity(1).
    Cards normalized (card+1)/28; -1 -> 0.
    """
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

# --- Define the Policy Network ---

# Which pair of indices (out of 5) to keep when discarding: (0,1), (0,2), ..., (3,4)
KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)  # 10


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_action_types=5, num_raise_classes=100, num_discard_classes=NUM_DISCARD_CLASSES):
        """
        New variant: shared base and three heads.
          - Action type: 5 actions (FOLD, RAISE, CHECK, CALL, DISCARD)
          - Raise: 100 classes -> [1, 100]
          - Discard: 10 classes -> which pair (i,j) of 5 hole cards to keep
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head = nn.Linear(hidden_dim, num_discard_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_type_logits = self.action_type_head(x)
        raise_logits = self.raise_head(x)
        discard_logits = self.discard_head(x)
        return action_type_logits, raise_logits, discard_logits

# --- Define the RL Agent using REINFORCE ---

class RLAgent:
    def __init__(self, input_dim, hidden_dim=128, lr=1e-3):
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = 0.99

    def select_action(self, state, valid_actions, min_raise, max_raise):
        """
        Sample an action. Returns (action_type, raise_amount, keep1, keep2), log_prob.
        For DISCARD, keep1 and keep2 are the two indices (0-4) to keep from 5 hole cards.
        """
        action_type_logits, raise_logits, discard_logits = self.policy_net(state)
        mask = (valid_actions == 0)
        masked_logits = action_type_logits.clone()
        masked_logits[mask] = -1e9

        action_type_dist = torch.distributions.Categorical(logits=masked_logits)
        raise_dist = torch.distributions.Categorical(logits=raise_logits)
        discard_dist = torch.distributions.Categorical(logits=discard_logits)

        action_type = action_type_dist.sample()
        raise_amount = raise_dist.sample()
        discard_idx = discard_dist.sample()

        log_prob = (action_type_dist.log_prob(action_type) +
                    raise_dist.log_prob(raise_amount) +
                    discard_dist.log_prob(discard_idx))

        action_type = action_type.item()
        raise_amount = raise_amount.item() + 1
        if action_type == PokerEnv.ActionType.RAISE.value:
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
        else:
            raise_amount = 0

        if action_type == PokerEnv.ActionType.DISCARD.value:
            keep1, keep2 = KEEP_PAIRS[discard_idx.item() % NUM_DISCARD_CLASSES]
        else:
            keep1, keep2 = 0, 0

        return (action_type, raise_amount, keep1, keep2), log_prob

    def update_policy(self, trajectory):
        R = 0
        returns = []
        for _, r in reversed(trajectory):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if returns.std() > 1e-5:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        else:
            returns = returns - returns.mean()
        loss = 0
        for (log_prob, _), R in zip(trajectory, returns):
            loss += -log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

import copy

class HistoricalRLAgent(Agent):
    """
    过去的幽灵：冻结了某一个历史时刻权重的 RL Agent。
    只负责凭直觉打牌（前向传播），绝对不更新权重（不学习）。
    """
    def __init__(self, state_dict, device):
        super().__init__(stream=False)
        self.device = device
        self.policy_net = PolicyNetwork(input_dim=INPUT_DIM)
        # 加载历史权重
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.to(device)
        self.policy_net.eval() # 冻结大脑，进入实战模式

    def __name__(self):
        return "HistoricalRLAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        state = preprocess_observation(observation).to(self.device)
        valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.float32, device=self.device)

        with torch.no_grad(): # 绝对不计算梯度
            action_type_logits, raise_logits, discard_logits = self.policy_net(state)

            mask = (valid_actions_tensor == 0)
            action_type_logits = action_type_logits.clone()
            action_type_logits[mask] = -1e9

            action_type = torch.distributions.Categorical(logits=action_type_logits).sample().item()

            if action_type == PokerEnv.ActionType.RAISE.value:
                raise_amount = torch.distributions.Categorical(logits=raise_logits).sample().item() + 1
                raise_amount = int(max(min(raise_amount, max_raise), min_raise))
            else:
                raise_amount = 0

            if action_type == PokerEnv.ActionType.DISCARD.value:
                discard_idx = torch.distributions.Categorical(logits=discard_logits).sample().item()
                keep1, keep2 = KEEP_PAIRS[discard_idx % NUM_DISCARD_CLASSES]
            else:
                keep1, keep2 = 0, 0

        return (action_type, raise_amount, keep1, keep2)

# --- Training Loop with Opponent Agent, CUDA support, and Weight Saving ---

def train_agent(num_episodes=20000, save_every=500, snapshot_every=1000, weight_path=None):
    if weight_path is None:
        weight_path = os.path.join(os.path.dirname(__file__), "agents", "rl_agent_weights.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 系统点火！使用设备: {device}")
    
    env = PokerEnv()
    agent = RLAgent(input_dim=INPUT_DIM)
    agent.policy_net.to(device)
    
    # 🥊 初始化初始角斗场（包含 5 种基础门派）
    base_pool = [
        ProbabilityAgent(),
        FoldAgent(),
        CallingStationAgent(),
        AllInAgent(),
        RandomAgent()
    ]
    historical_pool = [] # 历史幽灵池（初始为空）
    
    for episode in range(num_episodes):
        # 📸 快照机制：每过 snapshot_every 局，把当前的自己克隆并扔进幽灵池
        if episode > 0 and episode % snapshot_every == 0:
            print(f"📸 抽取灵魂快照！将第 {episode} 局的自己加入幽灵池！")
            cloned_state = copy.deepcopy(agent.policy_net.state_dict())
            ghost = HistoricalRLAgent(cloned_state, device)
            historical_pool.append(ghost)
            
            # 为了防止老幽灵太多导致没空打基础 Agent，控制幽灵池上限为 10 个（淘汰最老的）
            if len(historical_pool) > 10:
                historical_pool.pop(0)
                
        # 🎲 每局开始前，从“基础池”和“幽灵池”的混合体中随机挑一个对手
        current_pool = base_pool + historical_pool
        current_opponent = random.choice(current_pool)

        obs, info = env.reset()
        trajectory = []  
        total_reward = 0
        done = False

        while not done:
            acting_agent = obs[0]["acting_agent"]
            if acting_agent == 0:
                state = preprocess_observation(obs[0]).to(device)
                valid_actions_tensor = torch.tensor(obs[0]["valid_actions"], dtype=torch.float32, device=device)
                min_raise_val = obs[0]["min_raise"]
                max_raise_val = obs[0]["max_raise"]
                action, log_prob = agent.select_action(state, valid_actions_tensor, min_raise_val, max_raise_val)
                our_turn = True
            else:
                action = current_opponent.act(obs[1], reward=0, terminated=False, truncated=False, info={})
                our_turn = False
                log_prob = None

            obs, reward, done, truncated, info = env.step(action)
            r = reward[0]  
            total_reward += r
            if our_turn:
                trajectory.append((log_prob, r))

        agent.update_policy(trajectory)
        
        # 精简日志：只打印整百局的进度，或者加入幽灵时的提示，防止终端刷屏卡死
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Vs {current_opponent.__name__()} | Total Reward: {total_reward:.2f} | League Size: {len(current_pool)}")

        if (episode + 1) % save_every == 0:
            torch.save(agent.policy_net.state_dict(), weight_path)

    torch.save(agent.policy_net.state_dict(), weight_path)
    print(f"🎉 训练完成！最终 GTO 权重已保存至 {weight_path}")

if __name__ == "__main__":
    train_agent(num_episodes=1000000, save_every=5000, snapshot_every=5000)
