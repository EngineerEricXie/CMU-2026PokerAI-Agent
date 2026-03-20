import json
import random
import itertools
from collections import Counter
from functools import lru_cache
import time
import pickle  # ### 新增：用來讀取字典 ###
import os      # ### 新增 ###

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval

# 引入官方动作和卡牌翻译官
ActionType = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
int_card_to_str = PokerEnv.int_card_to_str # ### 新增：用於解析花色 ###

# 初始化官方牌力评估神器
evaluator = WrappedEval()
# 新增全域字典變數
GLOBAL_LOOKUP_TABLE = None
GLOBAL_PREFLOP_TABLE = None
GLOBAL_EHS_TABLE = None

# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------
def valid_cards(card_tuple) -> list:
    return [c for c in card_tuple if c >= 0]

# ---------------------------------------------------------------------------
# Hand evaluator (Using Official Engine with Cache)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=20000)
def cached_evaluate_best_hand(all_cards_tuple) -> int:
    best_score = float('inf')
    for combo in itertools.combinations(all_cards_tuple, 5):
        score = evaluator.evaluate(list(combo), [])
        if score < best_score:
            best_score = score
    return best_score

def evaluate_best_hand(hole_str: list, board_str: list) -> int:
    all_cards_tuple = tuple(sorted(hole_str + board_str))
    return cached_evaluate_best_hand(all_cards_tuple)

# ---------------------------------------------------------------------------
# Monte Carlo simulation (保留用於下注時的勝率評估)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def exact_hand_equity(my_cards_tuple, community_tuple, opp_discarded_tuple, my_discarded_tuple):
    """
    取代原本的蒙地卡羅！
    利用 pre-compute 字典進行 100% 精準的真實勝率窮舉，計算速度極快且零誤差。
    """
    global GLOBAL_LOOKUP_TABLE
    if GLOBAL_LOOKUP_TABLE is None:
         return 0.5 # 安全防護
    known_cards = set(my_cards_tuple) | set(community_tuple) | set(opp_discarded_tuple) | set(my_discarded_tuple)
    unknown_cards = [c for c in range(27) if c not in known_cards]
    
    board_needed = 5 - len(community_tuple)
    wins = ties = total = 0
    
    # 1. 窮舉對手可能的手牌 (從未知的牌中抽 2 張)
    for opp_hole in itertools.combinations(unknown_cards, 2):
        # 剩下的牌庫
        rem_deck = [c for c in unknown_cards if c not in opp_hole]
        
        # 2. 窮舉未來可能發出的公牌 (Turn / River)
        # 如果是 River (board_needed=0)，這個 itertools.combinations 會回傳一個空的 tuple，剛好跑 1 次
        for future_comm in itertools.combinations(rem_deck, board_needed):
            # 組裝雙方的 7 張牌
            my_7 = tuple(sorted(my_cards_tuple + community_tuple + future_comm))
            opp_7 = tuple(sorted(opp_hole + community_tuple + future_comm))
            
            # O(1) 瞬間查表！
            my_score = GLOBAL_LOOKUP_TABLE[my_7]
            opp_score = GLOBAL_LOOKUP_TABLE[opp_7]
            
            # treys 分數越小越強
            if my_score < opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
            total += 1
            
    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total

# 封裝函數以相容你原本的寫法
def get_exact_strength(my_cards: list, community: list, opp_discarded: list, my_discarded: list) -> float:
    # 這裡的 lookup_table 不能被 cache hash，我們用一個小技巧，把它當 id() 或不放進 cache 
    # 最簡單的方法是直接讓這個包裹函數處理轉換
    return exact_hand_equity(
        tuple(my_cards), 
        tuple(community), 
        tuple(opp_discarded), 
        tuple(my_discarded) # 注意：lru_cache 遇到 dict 會報錯，實務上建議把 table 設為全域變數，或用 class 包裝
    )

# ===========================================================================
# ### 新增：窮舉法棄牌決策迴圈 (Exhaustive Search Discard) ###
# ===========================================================================
def exhaustive_choose_discard(hole: list, community: list, opp_discarded: list) -> tuple:
    """利用預算好的 88 萬筆字典，瞬間算出期望值最高的保留組合"""
    global GLOBAL_LOOKUP_TABLE # 宣告使用全域變數
    best_idx = (0, 1)
    best_avg_score = float('inf') # treys 分數越小越強，所以找最小值

    known_cards = set(hole) | set(community) | set(opp_discarded)
    unknown_cards = [c for c in range(27) if c not in known_cards]

    # 針對 10 種保留組合進行評估
    for i, j in itertools.combinations(range(5), 2):
        kept = [hole[i], hole[j]]
        discarded = [hole[k] for k in range(5) if k not in (i, j)]
        
        total_score = 0
        combinations_count = 0
        
        # 窮舉未來的 2 張公牌 (Turn & River)
        for future_comm in itertools.combinations(unknown_cards, 2):
            # 組成 7 張牌，排序後直接查表！O(1) 的極致速度！
            final_7 = tuple(sorted(kept + community + list(future_comm)))
            total_score += GLOBAL_LOOKUP_TABLE[final_7]
            combinations_count += 1
            
        avg_score = total_score / combinations_count
        
        # --- 資訊洩漏懲罰 (Information Leakage Penalty) ---
        # TODO: 可能不用太嚴重，先觀察實戰效果再調整
        # 如果你丟棄的 3 張牌同花色，對手就知道你很難湊同花。我們給予「加分懲罰」讓這個選項變糟
        discarded_suits = [int_card_to_str(c)[1] for c in discarded]
        suit_counts = Counter(discarded_suits)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        if max_suit_count == 3:
            avg_score += 300  # 丟 3 張同花色，嚴重洩漏，重罰
        elif max_suit_count == 2:
            avg_score += 100  # 丟 2 張同花色，輕罰
            
        # 記錄平均分數最低（牌力最強）的組合
        if avg_score < best_avg_score:
            best_avg_score = avg_score
            best_idx = (i, j)
            
    return best_idx


# ---------------------------------------------------------------------------
# Player Agent
# ---------------------------------------------------------------------------
class PlayerAgent(Agent):

    def __init__(self, stream: bool = True):
        
        super().__init__(stream)
        self.action_types = ActionType
        self.start_time = time.perf_counter()
        self.time_limit = 450.0 
        self.total_hands = 1000
        self.net_chips = 0.0  
        self.is_guaranteed_win = False
        self.my_total_think_time = 0.0
        self.cfr_strategy = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(os.path.join(current_dir, "cfr_strategy.json"), "r") as f:
                self.cfr_strategy = json.load(f)
            self.logger.info(f"✅ 成功载入 CFR 完美策略表！包含 {len(self.cfr_strategy)} 个决策节点。")
        except Exception as e:
            self.logger.error(f"读取 CFR 策略表失败: {e}")
        
        # ### 新增：啟動時將 88 萬筆字典載入記憶體 ###
        global GLOBAL_LOOKUP_TABLE
        global GLOBAL_PREFLOP_TABLE
        global GLOBAL_EHS_TABLE
       
        try:
            with open(os.path.join(current_dir, "lookup_table_7cards.pkl"), "rb") as f:
                GLOBAL_LOOKUP_TABLE = pickle.load(f) 
            self.logger.info("✅ 成功載入 7 張牌力字典！棄牌將啟動窮舉引擎。")
        except Exception as e:
            self.logger.error(f"讀取字典失敗: {e}")

        # 讀取 Pre-flop 字典
        try:
            with open(os.path.join(current_dir, "preflop_table.pkl"), "rb") as f:
                GLOBAL_PREFLOP_TABLE = pickle.load(f) 
            self.logger.info("✅ 成功載入 Pre-flop 勝率表！")
        except Exception as e:
            self.logger.error(f"讀取 Pre-flop 字典失敗: {e}")
        
        try:
            with open(os.path.join(current_dir, "EHS_table_fixed.pkl"), "rb") as f:
                GLOBAL_EHS_TABLE = pickle.load(f) 
            # --- 新增：格式抽樣驗證 ---
            if GLOBAL_EHS_TABLE:
                sample_key = next(iter(GLOBAL_EHS_TABLE))
                # 驗證 Key 是否為 2 層 tuple，且包含底牌(2張)和公牌(3或4張)
                if isinstance(sample_key, tuple) and len(sample_key) == 2 and len(sample_key[0]) == 2:
                    self.logger.info(f"✅ EHS 雷達上線！格式驗證成功 (範例 Key: {sample_key})")
                else:
                    self.logger.error(f"❌ 警告：EHS 字典格式錯誤！預期 ((h1,h2), (c1,c2...))，實際拿到 {sample_key}")
                    GLOBAL_EHS_TABLE = None # 強制停用錯誤的表
            else:
                 self.logger.warning("⚠️ EHS 字典是空的！")
        except Exception as e:
            self.logger.error(f"讀取 EHS 字典失敗: {e}")

    def __name__(self):
        return "PlayerAgent"
    
    # ---------------------------------------------------------------------------
# 将复杂的牌局状态压缩成离散的信息集 ID (Infoset Key)
# ---------------------------------------------------------------------------
    def get_infoset_key(self, street: int, ehs_val: float, call_amount: int, pot_size: int) -> str:
        """
        将当前复杂的牌局状态，压缩成一个离散的字符串（信息集 ID）。
        CFR 将针对这些 ID 学习最优的 Fold/Call/Raise 概率分布。
        """
        # 1. 牌力分桶 (Hand Strength Bucket)
        # 基于原代码的 adj 阈值进行了平滑切分
        if ehs_val >= 0.75: hs_bucket = "Nuts" # 绝对强牌 (原代码 adj > 0.75)
        elif ehs_val >= 0.60: hs_bucket = "Strong" # 强牌/高潜听牌 (原代码 adj > 0.60)
        elif ehs_val >= 0.40: hs_bucket = "Marginal" # 边缘牌 (原代码 adj > 0.42)
        else: hs_bucket = "Trash" # 垃圾牌

        # 2. 下注压力/底池赔率分桶 (Betting Pressure Bucket)
        # 衡量我们要跟注的代价有多大
        if call_amount == 0: pressure = "None" # 可以免费 Check
        else:
            # 计算底池赔率
            pot_odds = call_amount / (pot_size + call_amount + 1e-9)
            if pot_odds < 0.20: pressure = "Low" # 便宜，随时可跟注
            elif pot_odds < 0.35: pressure = "Med" # 正常加注压力
            else: pressure = "High" # 对方下了重注，面临极高方差风险 (原代码 call_amount > 25 的防卫盾)

        return f"S{street}_{hs_bucket}_{pressure}"

    def act(self, observation, reward, terminated, truncated, info):
        act_start_time = time.perf_counter()

        current_hand = info.get('hand_number', 1)
        hands_left = self.total_hands - current_hand + 1

        valid_actions = observation["valid_actions"]
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

        can_raise = valid_actions[ActionType.RAISE.value]
        can_call  = valid_actions[ActionType.CALL.value]
        can_check = valid_actions[ActionType.CHECK.value]

        def record_and_return(action_tuple):
            self.my_total_think_time += (time.perf_counter() - act_start_time)
            return action_tuple

        def make_raise(fraction: float):
            amt = int(min_raise + (max_raise - min_raise) * fraction)
            amt = max(min_raise, min(max_raise, amt))
            return record_and_return((ActionType.RAISE.value, amt, 0, 0))

        def fold():  return record_and_return((ActionType.FOLD.value,  0, 0, 0))
        def call():  return record_and_return((ActionType.CALL.value,  0, 0, 0))
        def check(): return record_and_return((ActionType.CHECK.value, 0, 0, 0))
        def do_discard(i, j):
            return record_and_return((ActionType.DISCARD.value, 0, i, j))

        # =========================================================================
        # 第一步：必勝鎖定 (最高優先級！)
        # =========================================================================
        if not self.is_guaranteed_win:
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            if self.net_chips > max_possible_loss:
                self.is_guaranteed_win = True  
                self.logger.info(f"狀態切換：必勝鎖定！淨賺 {self.net_chips} > 極限損失 {max_possible_loss}。永久進入龜縮模式。")

        if self.is_guaranteed_win:
            if valid_actions[ActionType.DISCARD.value]:
                return do_discard(0, 1)
            if valid_actions[ActionType.FOLD.value]:
                return fold()
            return check()

        # =========================================================================
        # 第三步：處理強制換牌 (使用 O(1) 字典窮舉取代 MC)
        # =========================================================================
        if valid_actions[ActionType.DISCARD.value]:
            if len(my_cards) == 5:
                # 判斷全域變數是否成功載入
                global GLOBAL_LOOKUP_TABLE
                if GLOBAL_LOOKUP_TABLE is not None:
                    # 移除 lookup_table 參數
                    i, j = exhaustive_choose_discard(my_cards, community, opp_discarded)
                else:
                    # 舊版退路 (為了避免當機，若沒讀到檔案就隨便丟)
                    i, j = 0, 1 
            else:
                i, j = 0, 1
            return do_discard(i, j)
        
        # =========================================================================
        # 第四步：Pre-flop (Street 0) 專屬極簡過渡邏輯
        # =========================================================================
        if street == 0:
            global GLOBAL_PREFLOP_TABLE
            
            # 如果有成功載入表，且手上有 5 張牌
            if GLOBAL_PREFLOP_TABLE and len(my_cards) == 5:
                # O(1) 瞬間查出這 5 張起手牌的真實勝率
                preflop_equity = GLOBAL_PREFLOP_TABLE[tuple(sorted(my_cards))]
                
                # 基於勝率進行決策 (閾值可微調)
                if preflop_equity > 0.55:
                    # 強牌：積極加注
                    if can_raise:
                        # 你可以寫成固定加注，或依據勝率動態加注
                        return make_raise(0.3) 
                    return call() if can_call else check()
                    
                elif preflop_equity > 0.38:
                    # 中等牌：跟注看翻牌
                    if call_amount > 20: # 對手下大注就跑
                        return fold()
                    if can_call:
                        return call()
                    return check() if can_check else fold()
                    
                else:
                    # 爛牌：盡量不玩
                    if can_check:
                        return check() # 不用錢就看看
                    if can_call and call_amount <= 2:
                        return call()
                    return fold()
            else:
                # 防呆：如果字典沒載入，就退回保守策略
                if call_amount > 4: return fold()
                if can_check: return check()
                if can_call: return call()
                return fold()
            
        # =========================================================================
        # 第五步：Post-flop (Street 1~3) CFR 查表极速决策
        # =========================================================================
        
        # 1. 获取准确的 EHS (这部分保留你原本写得很好的代码)
        if street in (1, 2) and GLOBAL_EHS_TABLE: 
            state_key = (tuple(sorted(my_cards)), tuple(sorted(community)))
            if state_key in GLOBAL_EHS_TABLE:
                HS, PPot, NPot = GLOBAL_EHS_TABLE[state_key]
            else:
                HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
        else:
            HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
            PPot, NPot = 0.0, 0.0

        ehs_val = HS + (1 - HS) * PPot - HS * NPot
        
        # 2. 获取当前状态的 Bucket ID
        infoset = self.get_infoset_key(street, ehs_val, call_amount, pot)
        
        # 3. 查表获取策略概率 (如果遇到意料之外的状态，默认稳健策略：遇加注则弃牌，否则过牌)
        strategy = self.cfr_strategy.get(infoset, {"FOLD": 0.8, "CALL": 0.2, "RAISE": 0.0} if call_amount > 0 else {"FOLD": 0.0, "CALL": 1.0, "RAISE": 0.0})
        
        # 4. 根据概率随机抽取动作 (核心：混合策略)
        actions = list(strategy.keys())
        probs = list(strategy.values())
        import numpy as np
        probs_array = np.array(probs)
        probs_array = probs_array / probs_array.sum()
        chosen_action = np.random.choice(actions, p=probs_array)
        
        self.logger.info(f"CFR Node: {infoset} | Strategy: {strategy} | Executing: {chosen_action}")

        # 5. 执行动作转换 (结合引擎规则)
        if chosen_action == "RAISE" and can_raise:
            # AI 决定加注！使用默认的半池或随机注码，只要不超出比赛上限即可
            raise_frac = random.uniform(0.3, 0.6) 
            return make_raise(raise_frac)
            
        elif chosen_action == "CALL":
            # 在没有下注压力时，CALL 实际上就是 CHECK
            if call_amount == 0 and can_check:
                return check()
            if can_call:
                return call()
                
        elif chosen_action == "FOLD":
            # 免费过牌时绝对不弃牌，这是最低级的失误
            if call_amount == 0 and can_check:
                return check()
            return fold()
            
        # 绝对兜底逻辑，防止因为无效动作被判负
        if can_check: return check()
        if can_call and call_amount <= max(2, int(pot * 0.15)): return call()
        return fold()

    def observe(self, observation, reward, terminated, truncated, info):
        if reward is not None and reward != 0.0:
            self.net_chips += reward
            self.logger.info(f"Getting current game reward : {reward}, Overall reward: {self.net_chips}")

        if terminated:
            current_hand = info.get('hand_number', 1)
            hands_left = self.total_hands - current_hand
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            self.logger.info(f"🏁 Hand {current_hand} finished | Total earnings: {self.net_chips} | Distance to locking win (Max possible loss): {max_possible_loss}")