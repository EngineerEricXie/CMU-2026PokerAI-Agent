import json
import random
import itertools
from collections import Counter
from functools import lru_cache
import time
import pickle  # ### 新增：用來讀取字典 ###
import os      # ### 新增 ###
import itertools

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

def exhaustive_choose_discard_optimized(hole: list, community: list, opp_discarded: list) -> tuple:
    """
    终极穷举弃牌法 (Exact Equity Discard)
    计算量: 10种保留 x 120种对手底牌 x 91种未来发牌 = 109,200 次查表。
    耗时: Python 原生环境下约 0.1 ~ 0.2 秒，完美契合 1.5秒/手 的限制！
    """
    global GLOBAL_LOOKUP_TABLE # 依赖预加载的 88 万笔 7 张牌字典
    
    best_idx = (0, 1)
    best_win_rate = -1.0  # 我们追求最高胜率，所以初始设为 -1

    # 1. 精确缩减牌库：吃透一切已知情报！
    known_cards = set(hole) | set(community) | set(opp_discarded)
    unknown_cards = [c for c in range(27) if c not in known_cards]

    # 2. 遍历你 5 张底牌保留 2 张的 10 种组合
    for i, j in itertools.combinations(range(5), 2):
        keep_2 = (hole[i], hole[j])
        discard_3 = tuple(hole[k] for k in range(5) if k not in (i, j))
        
        wins = 0
        ties = 0
        total_scenarios = 0
        
        # 3. 穷举对手可能的 2 张底牌 (从剩余未知牌中挑) C(16, 2) = 120种
        for opp_2 in itertools.combinations(unknown_cards, 2):
            rem_deck = [c for c in unknown_cards if c not in opp_2]
            
            # 4. 穷举未来 Turn 和 River 要发的 2 张公共牌 C(14, 2) = 91种
            for future_2 in itertools.combinations(rem_deck, 2):
                
                # 组装双发最终的 7 张牌，排序后直接查表
                my_7 = tuple(sorted(keep_2 + tuple(community) + future_2))
                opp_7 = tuple(sorted(opp_2 + tuple(community) + future_2))
                
                my_score = GLOBAL_LOOKUP_TABLE[my_7]
                opp_score = GLOBAL_LOOKUP_TABLE[opp_7]
                
                # 记住：Treys / WrappedEval 的分数越小，牌力越强！
                if my_score < opp_score:
                    wins += 1
                elif my_score == opp_score:
                    ties += 1
                total_scenarios += 1
                
        # 5. 计算绝对真实的胜率 (Expected Equity)
        win_rate = (wins + 0.5 * ties) / total_scenarios
        
        # --- 6. 🧠 高阶博弈：科学的信息泄露惩罚 (Info-Leak Penalty) ---
        # 惩罚不用扣分数，而是直接扣除“胜率百分比”。
        penalty = 0.0
        
        # 提取丢弃的 3 张牌的花色 (0=♦, 1=♥, 2=♠)
        discard_suits = [c // 9 for c in discard_3] 
        unique_suits = len(set(discard_suits))
        
        if unique_suits == 1:
            # 丢掉 3 张同花色：告诉对手你没同花。稍微扣除 1.5% 的胜率估值
            penalty += 0.015 
        elif unique_suits == 2:
            # 丢掉 2 张同花色：扣除 0.5% 的胜率估值
            penalty += 0.005
            
        final_eval = win_rate - penalty
        
        # 记录真实胜率最高的组合
        if final_eval > best_win_rate:
            best_win_rate = final_eval
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
        self.my_id = None # 初始未知
        # --- 剥削引擎追踪器 ---
        self.hands_played = 0
        self.opp_folds = 0
        self.exploit_mode = False  # 是否处于剥削模式
        self.recent_profits = []   # 追踪最近 X 手的盈亏，用于动态回调

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
    def get_infoset_key(self, player_id, street, ehs_val, call_amount, pot_size):
        """同步 Trainer 的 Key 生成逻辑"""
        if ehs_val >= 0.75: hs = "Nuts"
        elif ehs_val >= 0.60: hs = "Strong"
        elif ehs_val >= 0.40: hs = "Marginal"
        else: hs = "Trash"

        odds = call_amount / (pot_size + call_amount + 1e-9)
        if call_amount == 0: pressure = "None"
        elif odds < 0.20: pressure = "Low"
        elif odds < 0.35: pressure = "Med"
        else: pressure = "High"

        return f"P{player_id}_S{street}_{hs}_{pressure}"

    def act(self, observation, reward, terminated, truncated, info):
        act_start_time = time.perf_counter()
        if self.my_id is None:
            self.my_id = observation["acting_agent"] 
            self.logger.info(f"身份确认：我是 Player {self.my_id} ({'SB' if self.my_id==0 else 'BB'})")

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
                    # 调用优化后的最新穷举函数
                    i, j = exhaustive_choose_discard_optimized(my_cards, community, opp_discarded)
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
        infoset = self.get_infoset_key(self.my_id, street, ehs_val, max(0, call_amount), pot)
        
        # 3. 查表获取策略概率 (适配最新的 CHECK 和 CALL)
        # 如果对手下注了，兜底防守是 CALL；如果免费，兜底防守是 CHECK
        fallback_strategy = {"CALL": 1.0} if call_amount > 0 else {"CHECK": 1.0}
        base_strategy = self.cfr_strategy.get(infoset, fallback_strategy)
        # 将 GTO 基础策略送入剥削引擎进行扭曲！
        strategy = self.apply_exploitative_adjustment(base_strategy, ehs_val, max(0, call_amount))
        
        # 4. 根据概率随机抽取动作 (核心：混合策略)
        
        # 4. 根据概率随机抽取动作 (核心：混合策略)
        actions = list(strategy.keys())
        probs = list(strategy.values())
        import numpy as np
        # 加上 1e-5 的极小扰动，防止所有概率为 0 导致崩溃
        probs_array = np.array(probs, dtype=np.float64) + 1e-5
        probs_array = probs_array / probs_array.sum()
        chosen_action = np.random.choice(actions, p=probs_array)
        
        self.logger.info(f"CFR Node: {infoset} | Strategy: {strategy} | Executing: {chosen_action}")

        # 5. 执行动作转换 (结合引擎规则)
        if chosen_action == "RAISE" and can_raise:
            # AI 决定加注！使用 CFR 的意图，在合理范围内重拳出击
            raise_frac = random.uniform(0.3, 0.6) 
            return make_raise(raise_frac)
            
        elif chosen_action == "CHECK" and can_check:
            # 精确响应 CHECK 指令
            return check()
            
        elif chosen_action in ["CALL", "RAISE"]:
            # 智能降级：如果指令是 CALL，或者想 RAISE 但达到了比赛上限不合法时，稳妥地收下筹码
            if can_call:
                return call()
            elif can_check: # 万一引擎状态异常，不要弃牌，免费看牌
                return check()
                
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
        # 记录单手盈亏
        if reward is not None and reward != 0.0:
            self.net_chips += reward
            self.logger.info(f"Hand reward: {reward}, Net chips: {self.net_chips}")

        if terminated:
            self.hands_played += 1
            current_hand = info.get('hand_number', 1)
            hands_left = self.total_hands - current_hand
            
            # --- 对手建模：统计弃牌率 ---
            # 如果在 River 摊牌前游戏就结束了，且我们赢了，说明对手 Fold 了
            if observation["street"] <= 3 and reward > 0:
                self.opp_folds += 1
                
            # 记录最近 20 手的盈亏（用于检测剥削是否失败）
            self.recent_profits.append(reward)
            if len(self.recent_profits) > 20:
                self.recent_profits.pop(0)
                
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            self.logger.info(f"🏁 Hand {current_hand} | Opp Fold Rate: {(self.opp_folds/self.hands_played):.1%} | Net: {self.net_chips} | Max Loss: {max_possible_loss}")

    def apply_exploitative_adjustment(self, strategy, true_ehs, call_amount):
        """
        动态扭曲 GTO 概率，对极端对手进行剥削
        """
        # 样本量太小，或者已经进入必胜锁定模式，不进行剥削，保持纯 GTO
        if self.hands_played < 50 or self.is_guaranteed_win:
            self.exploit_mode = False
            return strategy

        opp_fold_rate = self.opp_folds / self.hands_played
        
        # 动态回调 (Safety Net)：如果我们最近 20 手在亏钱，说明对手适应了或者我们在方差里，立刻缩回 GTO 龟壳！
        if sum(self.recent_profits) < -10:
            if self.exploit_mode:
                self.logger.warning("🛡️ 警报：剥削策略失效，对手可能在反击！立刻退回绝对 GTO 防御！")
                self.exploit_mode = False
            return strategy

        new_strategy = strategy.copy()

        # ---------------------------------------------------------
        # 剥削画像 A：软弱的“弃牌机器” (Nit)
        # 对手极度害怕加注，弃牌率异常高 (> 55%)
        # 策略：疯狂增加诈唬 (Bluff) 频率，用垃圾牌偷他的盲注！
        # ---------------------------------------------------------
        if opp_fold_rate > 0.55:
            self.exploit_mode = True
            if true_ehs < 0.40 and "RAISE" in new_strategy: # 拿着烂牌
                # 把 Fold 的概率转移到 Raise 上，强行诈唬
                if "FOLD" in new_strategy and new_strategy["FOLD"] > 0.2:
                    stolen_prob = new_strategy["FOLD"] * 0.4 # 偷取 40% 的弃牌概率
                    new_strategy["FOLD"] -= stolen_prob
                    new_strategy["RAISE"] += stolen_prob
                    self.logger.debug("🔪 剥削触发：对手太怂，强行增加垃圾牌诈唬频率！")

        # ---------------------------------------------------------
        # 剥削画像 B：头铁的“跟注站” (Calling Station)
        # 对手像疯狗一样几乎不弃牌 (< 25%)
        # 策略：关闭所有诈唬！有牌就往死里打价值，没牌就立刻跑！
        # ---------------------------------------------------------
        elif opp_fold_rate < 0.25:
            self.exploit_mode = True
            if true_ehs < 0.50: 
                # 烂牌绝对不诈唬，把 Raise 概率清零，全部转给 Fold 或 Check
                if "RAISE" in new_strategy and new_strategy["RAISE"] > 0:
                    transfer = new_strategy["RAISE"]
                    new_strategy["RAISE"] = 0.0
                    if call_amount > 0:
                        new_strategy["FOLD"] = new_strategy.get("FOLD", 0.0) + transfer
                    else:
                        new_strategy["CHECK"] = new_strategy.get("CHECK", 0.0) + transfer
                    self.logger.debug("🔪 剥削触发：对手跟注站，取消诈唬，没牌直接跑！")
            elif true_ehs > 0.70:
                # 拿到了好牌，把 Call/Check 的概率全转给 Raise，榨干他的血！
                if "CALL" in new_strategy and "RAISE" in new_strategy:
                    new_strategy["RAISE"] += new_strategy["CALL"] * 0.6
                    new_strategy["CALL"] *= 0.4
                    self.logger.debug("🔪 剥削触发：对手跟注站，拿到好牌疯狂加注榨取价值！")

        return new_strategy