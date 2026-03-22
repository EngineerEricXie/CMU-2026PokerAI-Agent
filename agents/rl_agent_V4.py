# ==========================================
# RLAgent V4 - 混合神经引擎版本 (EHS + RL + Anti-Exploit)
# ==========================================
# 1. 继承 V3 的 O(1) 极速弃牌引擎，通过“循环倒置”彻底解决 Timeout 超时瓶颈。
# 2. 引入推断期温度缩放 (Temperature Scaling) 机制，重塑 RL 动作概率分布。
# 3. 针对性治愈 RL 的“模式崩溃 (Mode Collapse)”，彻底抹除死板的“60/38”下注指纹。
# 4. 强行激活混合策略 (Mixed Strategy)，恢复 RL 大脑在转牌 (Turn) 和河牌 (River) 的下注与诈唬勇气。
# 5. 构建全域“反剥削 (Anti-Exploit) 装甲”，让专门针对固定下注金额的死板脚本 (如 Parkway Gardens) 彻底失效。

# ==========================================
# RLAgent V3 - 混合神经引擎版本 (EHS + RL)
# ==========================================
# 1. 优化了弃牌loop的速度


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
# 注释掉的地方用的是agent里生成的weight，它在随着训练一直更新。因此在这里暂时改为submission里的weight，它是固定的
# WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rl_agent_weights.pth") 
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "rl_agent_weights.pth") 

# ==========================================
# 1. 全域字典与辅助函数 (O(1) 完美算力引擎)
# ==========================================
GLOBAL_LOOKUP_TABLE = None
GLOBAL_PREFLOP_TABLE = None

def valid_cards(card_tuple) -> list:
    return [c for c in card_tuple if c >= 0]

def exhaustive_choose_discard_optimized(hole: list, community: list, opp_discarded: list) -> tuple:
    """
    【10倍提速版】完美弃牌引擎：循环倒置，消除重复计算
    """
    global GLOBAL_LOOKUP_TABLE
    if GLOBAL_LOOKUP_TABLE is None: return (0, 1)
    
    known_cards = set(hole) | set(community) | set(opp_discarded)
    unknown_cards = [c for c in range(27) if c not in known_cards]

    keep_pairs = list(itertools.combinations(range(5), 2))
    keep_2_list = [(hole[i], hole[j]) for i, j in keep_pairs]
    
    wins = [0] * 10
    ties = [0] * 10
    total = 0
    comm_tup = tuple(community)

    # 🚀 核心优化：先枚举未来的公共牌，再枚举对手的牌
    for future_2 in itertools.combinations(unknown_cards, 2):
        board_5 = comm_tup + future_2
        
        # 在这个特定的牌面下，提前算出我们 10 种留牌方案的绝对牌力
        my_scores = [GLOBAL_LOOKUP_TABLE[tuple(sorted(k2 + board_5))] for k2 in keep_2_list]
        
        rem_deck = [c for c in unknown_cards if c not in future_2]
        for opp_2 in itertools.combinations(rem_deck, 2):
            opp_score = GLOBAL_LOOKUP_TABLE[tuple(sorted(opp_2 + board_5))]
            
            for idx in range(10):
                if my_scores[idx] < opp_score: wins[idx] += 1
                elif my_scores[idx] == opp_score: ties[idx] += 1
            total += 1
            
    best_idx = (0, 1)
    best_win_rate = -1.0
    
    for idx in range(10):
        win_rate = (wins[idx] + 0.5 * ties[idx]) / total if total > 0 else 0.5
        i, j = keep_pairs[idx]
        discard_3 = tuple(hole[k] for k in range(5) if k not in (i, j))
        
        # 信息泄露惩罚
        discard_suits = [c // 9 for c in discard_3] 
        unique_suits = len(set(discard_suits))
        penalty = 0.015 if unique_suits == 1 else (0.005 if unique_suits == 2 else 0.0)
        
        final_score = win_rate - penalty
        if final_score > best_win_rate:
            best_win_rate = final_score
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
        # 🧠 第四层博弈：RL 神经网络接管下注博弈 (带温度缩放)
        # --------------------------------------------------------
        state = preprocess_observation(observation).to(self.device)
        valid_actions_tensor = torch.tensor(valid_actions, dtype=torch.float32).to(self.device)
        
        # 🌡️ 温度参数配置
        # T > 1.0 会让概率分布变平缓 (增加随机性，防剥削)
        # T < 1.0 会让概率分布变陡峭 (更加贪婪)
        ACTION_TEMP = 1.5   # 让 Fold/Check/Raise 的选择更多样
        RAISE_TEMP = 5.0    # 大幅打破 60/38 的刻板印象，让下注额变幻莫测

        with torch.no_grad():
            action_type_logits, raise_logits, _ = self.policy_net(state)

            mask = valid_actions_tensor == 0
            action_type_logits = action_type_logits.clone()
            
            # 1. 动作类型加入温度
            action_type_logits = action_type_logits / ACTION_TEMP
            action_type_logits[mask] = -1e9 # 掩码必须在最后，确保非法动作依然是绝对的不可能

            # 使用加入了温度的概率分布进行抽样
            action_type = torch.distributions.Categorical(logits=action_type_logits).sample().item()

            if action_type == self.action_types.RAISE.value:
                # 2. 下注额度加入极高温度，彻底融化固定的 60/38
                scaled_raise_logits = raise_logits / RAISE_TEMP
                raise_amount = torch.distributions.Categorical(logits=scaled_raise_logits).sample().item() + 1
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