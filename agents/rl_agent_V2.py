# ==========================================
# RLAgent V2 - 混合神经引擎版本 (EHS + RL)
# ==========================================
# 1. 结合了 EHS 完美弃牌引擎和 RL 神经网络的混合策略
# 2. 引入了锦标赛必胜锁机制，确保在赢够了之后进入绝对防守模式
# 3. 预加载了 O(1) 完美弃牌字典和 Pre-flop 牌力字典，提升决策质量
# 4. RL 大脑专注于下注决策，弃牌决策由完美引擎主导
# 5. 适当调整了弃牌策略的信息泄露惩罚，鼓励更多样化的弃牌组合
# 6. 训练了更大规模的 RL 模型，提升了下注决策的智能水平
# 7. 通过混合策略和多层防御机制，RLAgent V2 在锦标赛中表现更为稳健和强大，能够适应各种对手和牌局情况。
# 注意：本版本的 RLAgent 仍然使用了之前训练好的 RL 模型权重，但弃牌决策完全由完美引擎控制，确保在 Flop阶段的最优弃牌选择。
# 同时，RL 大脑专注于下注决策，利用训练中学到的策略进行博弈。锦标赛必胜锁定机制确保在赢够了之后进入绝对防守模式，最大化最终获胜的概率。


import os
import random
import torch
import pickle
import itertools
from agents.agent import Agent
from gym_env import PokerEnv


from train_rl_agent import (
    PolicyNetwork,
    preprocess_observation,
    INPUT_DIM,
    KEEP_PAIRS,
    NUM_DISCARD_CLASSES,
)

action_types = PokerEnv.ActionType
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rl_agent_weights.pth")

# ==========================================
# 1. 全域字典与辅助函数 (O(1) 完美算力引擎)
# ==========================================
GLOBAL_LOOKUP_TABLE = None
GLOBAL_PREFLOP_TABLE = None

def valid_cards(card_tuple) -> list:
    return [c for c in card_tuple if c >= 0]

def exhaustive_choose_discard_optimized(hole: list, community: list, opp_discarded: list) -> tuple:
    global GLOBAL_LOOKUP_TABLE
    if GLOBAL_LOOKUP_TABLE is None: return (0, 1)
    
    best_idx = (0, 1)
    best_win_rate = -1.0
    known_cards = set(hole) | set(community) | set(opp_discarded)
    unknown_cards = [c for c in range(27) if c not in known_cards]

    for i, j in itertools.combinations(range(5), 2):
        keep_2 = (hole[i], hole[j])
        discard_3 = tuple(hole[k] for k in range(5) if k not in (i, j))
        wins = ties = total = 0
        
        for opp_2 in itertools.combinations(unknown_cards, 2):
            rem_deck = [c for c in unknown_cards if c not in opp_2]
            for future_2 in itertools.combinations(rem_deck, 2):
                my_7 = tuple(sorted(keep_2 + tuple(community) + future_2))
                opp_7 = tuple(sorted(opp_2 + tuple(community) + future_2))
                my_score = GLOBAL_LOOKUP_TABLE[my_7]
                opp_score = GLOBAL_LOOKUP_TABLE[opp_7]
                if my_score < opp_score: wins += 1
                elif my_score == opp_score: ties += 1
                total += 1
                
        win_rate = (wins + 0.5 * ties) / total
        
        # 信息泄露惩罚
        discard_suits = [c // 9 for c in discard_3] 
        unique_suits = len(set(discard_suits))
        penalty = 0.015 if unique_suits == 1 else (0.005 if unique_suits == 2 else 0.0)
        
        if (win_rate - penalty) > best_win_rate:
            best_win_rate = win_rate - penalty
            best_idx = (i, j)
            
    return best_idx

# ==========================================
# 2. RLAgent 主体 (混合神经引擎)
# ==========================================
class RLAgent(Agent):
    def __name__(self):
        return "RLAgent"

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = action_types
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 追踪器：用于比赛后期的必胜锁定 ---
        self.net_chips = 0.0
        self.is_guaranteed_win = False
        self.total_hands = 1000

        # --- 1. 挂载 RL 大脑 ---
        self.policy_net = PolicyNetwork(input_dim=INPUT_DIM, num_discard_classes=NUM_DISCARD_CLASSES)
        if os.path.exists(WEIGHTS_PATH):
            state = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
            try:
                self.policy_net.load_state_dict(state, strict=True)
                self.logger.info(f"✅ 成功加载 RL 大脑权重: {WEIGHTS_PATH}")
            except Exception as e:
                self.logger.warning(f"❌ 权重加载失败: {e}. 降级为随机策略。")
        else:
            self.logger.warning(f"❌ 找不到权重文件 {WEIGHTS_PATH}")
        self.policy_net.to(self.device)
        self.policy_net.eval() # 开启实战模式

        # --- 2. 挂载 O(1) 字典 ---
        global GLOBAL_LOOKUP_TABLE, GLOBAL_PREFLOP_TABLE
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(os.path.join(current_dir, "lookup_table_7cards.pkl"), "rb") as f:
                GLOBAL_LOOKUP_TABLE = pickle.load(f) 
            self.logger.info("✅ 7 张牌完美弃牌字典挂载完毕！")
        except Exception as e:
            self.logger.error(f"读取 7张牌字典 失败: {e}")

        try:
            with open(os.path.join(current_dir, "preflop_table.pkl"), "rb") as f:
                GLOBAL_PREFLOP_TABLE = pickle.load(f) 
            self.logger.info("✅ Pre-flop 字典挂载完毕！")
        except Exception as e:
            self.logger.error(f"读取 Pre-flop 字典 失败: {e}")

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]
        street = observation["street"]
        my_cards = valid_cards(observation["my_cards"])
        community = valid_cards(observation["community_cards"])
        opp_discarded = valid_cards(observation["opp_discarded_cards"])
        
        current_hand = info.get('hand_number', 1)
        hands_left = self.total_hands - current_hand + 1

        # --------------------------------------------------------
        # 🛡️ 第一层防御：锦标赛必胜锁定 (赢够了就绝对防守)
        # --------------------------------------------------------
        if not self.is_guaranteed_win:
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            if self.net_chips > max_possible_loss:
                self.is_guaranteed_win = True  
                self.logger.info(f"🏆 必胜锁定！净赚 {self.net_chips} > 极限损失 {max_possible_loss}。永久进入防守模式。")

        if self.is_guaranteed_win:
            if valid_actions[self.action_types.DISCARD.value]: return (self.action_types.DISCARD.value, 0, 0, 1)
            if valid_actions[self.action_types.FOLD.value]: return (self.action_types.FOLD.value, 0, 0, 0)
            return (self.action_types.CHECK.value, 0, 0, 0)

        # --------------------------------------------------------
        # 🧮 第二层算力：绝对完美的 Flop 强制弃牌
        # --------------------------------------------------------
        if valid_actions[self.action_types.DISCARD.value]:
            if len(my_cards) == 5 and GLOBAL_LOOKUP_TABLE is not None:
                i, j = exhaustive_choose_discard_optimized(my_cards, community, opp_discarded)
            else:
                i, j = 0, 1 
            return (self.action_types.DISCARD.value, 0, i, j)
            
        # --------------------------------------------------------
        # ⚡ 第三层算力：Pre-flop 安全过滤
        # --------------------------------------------------------
        if street == 0 and GLOBAL_PREFLOP_TABLE and len(my_cards) == 5:
            ehs = GLOBAL_PREFLOP_TABLE.get(tuple(sorted(my_cards)), 0.5)
            # 极品烂牌直接丢，不浪费 RL 大脑的思考空间
            if ehs < 0.40:
                return (self.action_types.CHECK.value, 0, 0, 0) if valid_actions[self.action_types.CHECK.value] else (self.action_types.FOLD.value, 0, 0, 0)

        # --------------------------------------------------------
        # 🧠 第四层博弈：RL 神经网络接管下注博弈
        # --------------------------------------------------------
        state = preprocess_observation(observation).to(self.device)
        valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_type_logits, raise_logits, _ = self.policy_net(state)

            mask = valid_actions_tensor == 0
            action_type_logits = action_type_logits.clone()
            action_type_logits[mask] = -1e9

            # 使用 RL 训练出的概率分布进行抽样（混合策略，对手无法预测！）
            action_type = torch.distributions.Categorical(logits=action_type_logits).sample().item()

            if action_type == self.action_types.RAISE.value:
                raise_amount = torch.distributions.Categorical(logits=raise_logits).sample().item() + 1
                raise_amount = int(max(min(raise_amount, max_raise), min_raise))
            else:
                raise_amount = 0

        return (action_type, raise_amount, 0, 0)

    def observe(self, observation, reward, terminated, truncated, info):
        if reward is not None and reward != 0.0:
            self.net_chips += reward
        if terminated:
            current_hand = info.get('hand_number', 1)
            self.logger.info(f"🏁 Hand {current_hand} finished | Net Chips: {self.net_chips}")