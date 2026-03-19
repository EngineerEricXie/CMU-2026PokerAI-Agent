import random
import itertools
from collections import Counter, deque
from functools import lru_cache
import time
import pickle
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval

ActionType = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
int_card_to_str = PokerEnv.int_card_to_str

evaluator = WrappedEval()

GLOBAL_LOOKUP_TABLE = None
GLOBAL_PREFLOP_TABLE = None

# ===========================================================================
# 1. 神經網路結構與經驗池
# ===========================================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 增加網路寬度，讓它更能理解複雜的特徵互動
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ===========================================================================
# 2. 原有 Card Helpers & 查表引擎 (保持不變)
# ===========================================================================
def valid_cards(card_tuple) -> list: return [c for c in card_tuple if c >= 0]

@lru_cache(maxsize=20000)
def cached_evaluate_best_hand(all_cards_tuple) -> int:
    best_score = float('inf')
    for combo in itertools.combinations(all_cards_tuple, 5):
        score = evaluator.evaluate(list(combo), [])
        if score < best_score: best_score = score
    return best_score

def evaluate_best_hand(hole_str: list, board_str: list) -> int:
    return cached_evaluate_best_hand(tuple(sorted(hole_str + board_str)))

@lru_cache(maxsize=2048)
def exact_hand_equity(my_cards_tuple, community_tuple, opp_discarded_tuple, my_discarded_tuple):
    global GLOBAL_LOOKUP_TABLE
    if GLOBAL_LOOKUP_TABLE is None: return 0.5 
    known_cards = set(my_cards_tuple) | set(community_tuple) | set(opp_discarded_tuple) | set(my_discarded_tuple)
    unknown_cards = [c for c in range(27) if c not in known_cards]
    
    board_needed = 5 - len(community_tuple)
    wins = ties = total = 0
    
    for opp_hole in itertools.combinations(unknown_cards, 2):
        rem_deck = [c for c in unknown_cards if c not in opp_hole]
        for future_comm in itertools.combinations(rem_deck, board_needed):
            my_7 = tuple(sorted(my_cards_tuple + community_tuple + future_comm))
            opp_7 = tuple(sorted(opp_hole + community_tuple + future_comm))
            
            my_score = GLOBAL_LOOKUP_TABLE[my_7]
            opp_score = GLOBAL_LOOKUP_TABLE[opp_7]
            
            if my_score < opp_score: wins += 1
            elif my_score == opp_score: ties += 1
            total += 1
            
    if total == 0: return 0.5
    return (wins + 0.5 * ties) / total

def get_exact_strength(my_cards: list, community: list, opp_discarded: list, my_discarded: list) -> float:
    return exact_hand_equity(tuple(my_cards), tuple(community), tuple(opp_discarded), tuple(my_discarded))

def exhaustive_choose_discard(hole: list, community: list, opp_discarded: list) -> tuple:
    global GLOBAL_LOOKUP_TABLE 
    best_idx, best_avg_score = (0, 1), float('inf')
    known_cards = set(hole) | set(community) | set(opp_discarded)
    unknown_cards = [c for c in range(27) if c not in known_cards]

    for i, j in itertools.combinations(range(5), 2):
        kept = [hole[i], hole[j]]
        discarded = [hole[k] for k in range(5) if k not in (i, j)]
        total_score, count = 0, 0
        
        for future_comm in itertools.combinations(unknown_cards, 2):
            final_7 = tuple(sorted(kept + community + list(future_comm)))
            total_score += GLOBAL_LOOKUP_TABLE[final_7]
            count += 1
            
        avg_score = total_score / count
        discarded_suits = [int_card_to_str(c)[1] for c in discarded]
        suit_counts = Counter(discarded_suits)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        if max_suit_count == 3: avg_score += 300 
        elif max_suit_count == 2: avg_score += 100 
            
        if avg_score < best_avg_score:
            best_avg_score = avg_score
            best_idx = (i, j)
    return best_idx

# ===========================================================================
# 3. DQN Agent 
# ===========================================================================
class PlayerAgent(Agent):

    def __init__(self, stream: bool = True, is_training: bool = False, start_from_new: bool = False, load_model_path: str = None):
        super().__init__(stream)
        self.action_types = ActionType
        self.start_time = time.perf_counter()
        self.time_limit = 450.0 
        self.total_hands = 1000
        self.net_chips = 0.0  
        self.is_guaranteed_win = False
        self.my_total_think_time = 0.0
        self.start_from_new = start_from_new
        self.load_model_path = load_model_path
        
        # 載入字典
        global GLOBAL_LOOKUP_TABLE, GLOBAL_PREFLOP_TABLE
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(os.path.join(current_dir, "lookup_table_7cards.pkl"), "rb") as f:
                GLOBAL_LOOKUP_TABLE = pickle.load(f) 
        except: pass
        
        try:
            with open(os.path.join(current_dir, "preflop_table.pkl"), "rb") as f:
                GLOBAL_PREFLOP_TABLE = pickle.load(f) 
        except: pass

        # --- DQN 初始化 ---
        self.is_training = is_training
        # 擴充特徵維度到 7，加入更多決策線索
        self.state_dim = 7 
        self.action_dim = 4 # 0:FOLD, 1:CALL/CHECK, 2:RAISE_MIN, 3:RAISE_POT
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        
        if not self.start_from_new:
            if self.load_model_path is None:
                model_path = os.path.join(current_dir, "models", "dqn_poker_final.pth")
            else:
                model_path = self.load_model_path
            self.load_model(model_path)
            
        
        if not self.is_training:
            self.epsilon = 0.01 
            self.q_net.eval()
        else:
            self.epsilon = 1.0
            
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999 # 放慢衰減速度，讓它探索更久
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0005) # 降低 LR 避免震盪
        self.memory = ReplayBuffer(capacity=100000)
        self.batch_size = 128
        self.gamma = 0.99
        
        # 🌟 新增：回合軌跡紀錄 (Episode Trajectory)
        self.current_episode_trajectory = []

    def __name__(self):
        return "DQNPokerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        act_start_time = time.perf_counter()
        current_hand = info.get('hand_number', 1)
        hands_left = self.total_hands - current_hand + 1

        my_cards      = valid_cards(observation["my_cards"])
        community     = valid_cards(observation["community_cards"])
        opp_discarded = valid_cards(observation["opp_discarded_cards"])
        my_discarded  = valid_cards(observation["my_discarded_cards"])
        my_bet        = observation["my_bet"]
        opp_bet       = observation["opp_bet"]
        min_raise     = observation["min_raise"]
        max_raise     = observation["max_raise"]
        street        = observation["street"]

        call_amount = opp_bet - my_bet
        pot         = my_bet + opp_bet

        def record_and_return(action_tuple):
            self.my_total_think_time += (time.perf_counter() - act_start_time)
            return action_tuple

        # --- 動作解析 ---
        valid_actions = observation["valid_actions"]
        can_fold, can_check, can_call, can_raise, can_discard = False, False, False, False, False
        
        if isinstance(valid_actions, dict):
            can_fold = valid_actions.get(ActionType.FOLD.value, valid_actions.get('FOLD', False))
            can_check = valid_actions.get(ActionType.CHECK.value, valid_actions.get('CHECK', False))
            can_call = valid_actions.get(ActionType.CALL.value, valid_actions.get('CALL', False))
            can_raise = valid_actions.get(ActionType.RAISE.value, valid_actions.get('RAISE', False))
            can_discard = valid_actions.get(ActionType.DISCARD.value, valid_actions.get('DISCARD', False))
        elif isinstance(valid_actions, (list, tuple)):
            if len(valid_actions) > ActionType.FOLD.value: can_fold = bool(valid_actions[ActionType.FOLD.value])
            if len(valid_actions) > ActionType.CHECK.value: can_check = bool(valid_actions[ActionType.CHECK.value])
            if len(valid_actions) > ActionType.CALL.value: can_call = bool(valid_actions[ActionType.CALL.value])
            if len(valid_actions) > ActionType.RAISE.value: can_raise = bool(valid_actions[ActionType.RAISE.value])
            if len(valid_actions) > ActionType.DISCARD.value: can_discard = bool(valid_actions[ActionType.DISCARD.value])

        # --- 必勝鎖定 ---
        if not self.is_training:
            if not self.is_guaranteed_win:
                max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
                if self.net_chips > max_possible_loss:
                    self.is_guaranteed_win = True 
                    self.logger.info(f"🎉 已鎖定勝利！目前籌碼: {self.net_chips:.2f}, 剩餘手數: {hands_left}, 最大可能損失: {max_possible_loss}")

            if self.is_guaranteed_win:
                self.logger.info(f"🔒 鎖定勝利，採取安全策略。當前籌碼: {self.net_chips:.2f}, 剩餘手數: {hands_left}")
                if can_discard: return record_and_return((ActionType.DISCARD.value, 0, 0, 1))
                if can_check: return record_and_return((ActionType.CHECK.value, 0, 0, 0))
                if can_fold: return record_and_return((ActionType.FOLD.value, 0, 0, 0))


        # --- 強制換牌階段 (不交給神經網路) ---
        if can_discard:
            global GLOBAL_LOOKUP_TABLE
            if len(my_cards) == 5 and GLOBAL_LOOKUP_TABLE is not None:
                i, j = exhaustive_choose_discard(my_cards, community, opp_discarded)
            else:
                i, j = 0, 1
            return record_and_return((ActionType.DISCARD.value, 0, i, j))

        # =========================================================
        # 🌟 神經網路特徵提取 (強化版)
        # =========================================================
        equity = 0.5
        if street == 0 and len(my_cards) == 5:
            global GLOBAL_PREFLOP_TABLE
            if GLOBAL_PREFLOP_TABLE:
                equity = GLOBAL_PREFLOP_TABLE.get(tuple(sorted(my_cards)), 0.5)
        else:
            equity = get_exact_strength(my_cards, community, opp_discarded, my_discarded)

        norm_street = street / 3.0
        # 底池賠率：如果 call_amount 為 0，賠率就是 0
        pot_odds = call_amount / (pot + call_amount + 1e-9)
        # 壓力指標：對方下注佔目前底池的比例
        bet_pressure = min(1.0, call_amount / (pot + 1e-9))
        # 籌碼深度指標：底池佔最大可能下注的比例 (表示這局打得多大)
        pot_depth = min(1.0, pot / 200.0) 
        
        # 動作遮罩特徵：讓網路知道現在能幹嘛
        mask_raise = 1.0 if can_raise else 0.0
        mask_call = 1.0 if (can_call or can_check) else 0.0

        current_state = np.array([
            norm_street, equity, pot_odds, bet_pressure, pot_depth, mask_raise, mask_call
        ], dtype=np.float32)

        # =========================================================
        # DQN 決策
        # =========================================================
        # 只有在有選擇空間的情況下才交給網路
        if not can_raise and not can_call and not can_check:
            return record_and_return((ActionType.FOLD.value, 0, 0, 0))

        if self.is_training and random.random() < self.epsilon:
            # 只在合法的動作中隨機選擇！
            available_actions = [0] # Fold 永遠可以選
            if can_call or can_check: available_actions.append(1)
            if can_raise: available_actions.extend([2, 3])
            action_idx = random.choice(available_actions)
        else:
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor).cpu().numpy()[0]
                
            # 應用無效動作懲罰 (Masking)：強迫網路不要選非法的動作
            if not can_raise:
                q_values[2] = -999.0
                q_values[3] = -999.0
            if not can_call and not can_check:
                q_values[1] = -999.0
                
            action_idx = np.argmax(q_values)

        # 記錄狀態到軌跡中 (此時還不知道 Reward)
        if self.is_training:
            self.current_episode_trajectory.append((current_state, action_idx))

        # --- 動作映射 ---
        if action_idx == 3 and can_raise: 
            return record_and_return((ActionType.RAISE.value, max_raise, 0, 0))
        if action_idx == 2 and can_raise: 
            amt = int(min_raise + (max_raise - min_raise) * 0.3) # RAISE_MIN 改成小加注
            return record_and_return((ActionType.RAISE.value, max(min_raise, min(max_raise, amt)), 0, 0))
        if action_idx == 1: 
            if can_check and call_amount == 0: return record_and_return((ActionType.CHECK.value, 0, 0, 0))
            if can_call: return record_and_return((ActionType.CALL.value, 0, 0, 0))
            
        return record_and_return((ActionType.FOLD.value, 0, 0, 0))
    
    def observe(self, observation, reward, terminated, truncated, info):
        if reward is not None and reward != 0.0:
            self.net_chips += reward

        done = terminated or truncated

        if self.is_training and done:
            # 🌟 回報分配 (Credit Assignment)：將最終贏/輸的籌碼，分配給這局所有的決策步驟！
            if len(self.current_episode_trajectory) > 0:
                # 最終狀態為全 0 向量
                terminal_state = np.zeros(self.state_dim, dtype=np.float32)
                
                # 遍歷軌跡，將最後的 reward 賦予每一步
                # 為了讓 AI 知道是「哪一步」導致的結果，可以使用 Gamma 折扣
                discounted_reward = reward
                for i in reversed(range(len(self.current_episode_trajectory))):
                    state, action_idx = self.current_episode_trajectory[i]
                    
                    # 只有最後一步的 next_state 是 terminal，前面的 next_state 是軌跡中的下一個 state
                    if i == len(self.current_episode_trajectory) - 1:
                        next_s = terminal_state
                        d = True
                    else:
                        next_s = self.current_episode_trajectory[i+1][0]
                        d = False
                        
                    self.memory.push(state, action_idx, discounted_reward, next_s, d)
                    discounted_reward *= self.gamma # 將回報往前傳遞
                
                # 清空軌跡
                self.current_episode_trajectory = []
                
                # 執行訓練
                self._train_step()

            # 衰減 Epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            current_hand = info.get('hand_number', 1)
            
            if current_hand % 10 == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                
            if current_hand % 100 == 0:
                self.logger.info(f"🧠 DQN Stats | Epsilon: {self.epsilon:.3f} | Memory: {len(self.memory)}")

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # 正規化 Reward，幫助神經網路穩定收斂！(假設單局籌碼變動在 -100 ~ 100)
        rewards = torch.FloatTensor(rewards / 100.0).unsqueeze(1).to(self.device) 
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止 Q 值爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
    def save_model(self, path="poker_dqn.pth"):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path="poker_dqn.pth"):
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.logger.info(f"✅ 成功載入預訓練模型 {path}")
            # self.is_training = False
            self.epsilon = 0.01
        else:
            self.logger.info(f"⚠️ 模型檔案 {path} 不存在，將從頭開始訓練。")