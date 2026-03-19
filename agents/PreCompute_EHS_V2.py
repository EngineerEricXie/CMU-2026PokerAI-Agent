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
class PC_EHS_V2(Agent):

    def __init__(self, stream: bool = True, is_training: bool = False):
        super().__init__(stream)
        self.action_types = ActionType
        self.start_time = time.perf_counter()
        self.time_limit = 450.0 
        self.total_hands = 1000
        self.net_chips = 0.0  
        self.is_guaranteed_win = False
        self.my_total_think_time = 0.0
        self.is_training = is_training
        
        # ### 新增：啟動時將 88 萬筆字典載入記憶體 ###
        global GLOBAL_LOOKUP_TABLE
        global GLOBAL_PREFLOP_TABLE
        global GLOBAL_EHS_TABLE
        current_dir = os.path.dirname(os.path.abspath(__file__))
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
        return "PC_EHS_V2"

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
        if not self.is_training:
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
                if preflop_equity > 0.65:
                    # 強牌：積極加注
                    if can_raise:
                        # 你可以寫成固定加注，或依據勝率動態加注
                        return make_raise(0.4) 
                    return call() if can_call else check()
                    
                elif preflop_equity > 0.5:
                    # 中等牌：跟注看翻牌
                    if call_amount > 10: # 對手下大注就跑
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
        # 第五步：Post-flop (Street 1~3) 绝对胜率精算与【动态风控下注引擎】
        # =========================================================================
        global GLOBAL_EHS_TABLE
        
        HS = 0.5
        PPot = 0.0
        NPot = 0.0
        
        # 1. 提取基础概率指标
        if street in (1, 2) and GLOBAL_EHS_TABLE: 
            state_key = (tuple(sorted(my_cards)), tuple(sorted(community)))
            if state_key in GLOBAL_EHS_TABLE:
                # HS, PPot, NPot = GLOBAL_EHS_TABLE[state_key]
                HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
            else:
                HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
        else:
            HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
            
        # 计算有效牌力 (Effective Hand Strength)
        ehs_val = HS + (1 - HS) * PPot - HS * NPot

        # 2. 计算环境变量与风险敞口
        pot_odds = call_amount / (pot + call_amount + 1e-9)
        # 对手下注的激烈程度 (0.0 到 1.0)
        bet_severity = call_amount / max_raise if max_raise > 0 else 0
        opp_is_all_in = (call_amount >= 90) # 是否是极限 All-in
        
        # =========================================================
        # 🛡️ 绝对风控防线 (Adverse Selection Shield) - 生死线
        # =========================================================
        if call_amount > 15:
            # 基础门槛是 Pot Odds，随注码飙升而指数级增加风险溢价
            dynamic_risk_premium = bet_severity * 0.40 
            safe_call_threshold = pot_odds + dynamic_risk_premium
            
            # 河牌圈纯拼大小，无听牌空间，阈值收紧
            if street == 3: 
                safe_call_threshold += 0.05 
                
            # 极限 All-in 护盾：强制要求绝对统治力
            if opp_is_all_in:
                safe_call_threshold = max(safe_call_threshold, 0.86)

            # 防护盾拦截判定
            if ehs_val < safe_call_threshold:
                # 日志记录拦截动作，方便复盘
                self.logger.debug(f"🛡️ 护盾拦截 | 敌注: {call_amount}, 牌力: {ehs_val:.2f} < 门槛: {safe_call_threshold:.2f}")
                return check() if can_check else fold()

        # =========================================================
        # ⚔️ 动态进攻与榨取逻辑 (Dynamic Value Extraction)
        # =========================================================
        # =========================================================
        # 👁️ 情報壓制詐唬 (Intel-based Bluff)
        # =========================================================
        if street in (1, 2) and opp_discarded:
            # 取得雙方花色資訊
            opp_discarded_suits = [int_card_to_str(c)[1] for c in opp_discarded]
            board_suits = [int_card_to_str(c)[1] for c in community]
            
            # 如果對手丟了 2 張或 3 張某花色，且公牌剛好在聽該花色或已成同花
            for suit in set(opp_discarded_suits):
                if opp_discarded_suits.count(suit) >= 2 and board_suits.count(suit) >= 2:
                    # 對手極高機率沒有同花，我們直接假裝我們有！
                    if can_raise and call_amount <= 20 and random.random() < 0.85:
                        self.logger.info("🎭 觸發情報壓制！對手棄掉關鍵花色，強力詐唬！")
                        return make_raise(0.60) # 打出 60% 底池的強力加注
        
        # 1. 绝对怪兽牌 (Monster Nuts)
        if ehs_val >= 0.94 or (HS >= 0.92 and NPot < 0.10):
            if can_raise:
                # 牌极大，不留情面，根据情况直接满注或重注
                raise_frac = min(1.0, 0.5 + (ehs_val - 0.9) * 5.0)
                return make_raise(raise_frac)
            return call() if can_call else check()

        # 2. 强价值牌 (Strong Value) - 赢面大但不能无脑 All-in
        elif ehs_val >= 0.84:
            if can_raise:
                # 如果对手很保守(没怎么下注)，我们打个半池逼取价值
                if call_amount < 15:
                    return make_raise(0.50)
                # 如果对手反抗激烈，仅控制底池或试探性小加注
                return make_raise(0.20) if random.random() < 0.5 else call()
            return call() if can_call else check()

        # 3. 高潜力听牌 (High Potential Draws) - 半诈唬
        elif HS < 0.55 and PPot >= 0.40 and street < 3:
            if can_raise and call_amount <= 20 and random.random() < 0.35: 
                return make_raise(0.35) # 试探性半诈唬施压
            if can_call and pot_odds <= 0.35: # 赔率合适就买梦
                return call()
            return check() if can_check else fold()

        # 4. 边缘底线牌 (Marginal Showdown Value)
        elif ehs_val >= 0.65:
            if can_check: 
                return check() 
            # 非常便宜的注（大概 1/5 底池）才勉强看牌
            if can_call and call_amount <= max(4, int(pot * 0.20)):
                return call()
            return fold()
            
        # =========================================================
        # 🎭 极简诈唬逻辑 (Simplified Bluff)
        # 仅在无压力、低损失、有加注权的情况下极低概率偷鸡
        # =========================================================
        if call_amount == 0 and can_raise and random.random() < 0.06:
            # 到了河牌偷鸡概率减半
            if street == 3 and random.random() > 0.5:
                return check()
            return make_raise(0.20 + random.uniform(-0.02, 0.05))
            
        # =========================================================
        # 🗑️ 默认安全降级
        # =========================================================
        return check() if can_check else fold()

    def observe(self, observation, reward, terminated, truncated, info):
        if reward is not None and reward != 0.0:
            self.net_chips += reward
            self.logger.info(f"Getting current game reward : {reward}, Overall reward: {self.net_chips}")

        if terminated:
            current_hand = info.get('hand_number', 1)
            hands_left = self.total_hands - current_hand
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            self.logger.info(f"🏁 Hand {current_hand} finished | Total earnings: {self.net_chips} | Distance to locking win (Max possible loss): {max_possible_loss}")