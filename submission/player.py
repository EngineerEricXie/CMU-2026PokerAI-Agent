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
        # discarded_suits = [int_card_to_str(c)[1] for c in discarded]
        # suit_counts = Counter(discarded_suits)
        # max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        # if max_suit_count == 3:
        #     avg_score += 3  # 丟 3 張同花色，嚴重洩漏，重罰
        # elif max_suit_count == 2:
        #     avg_score += 1  # 丟 2 張同花色，輕罰
            
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

        # ==========================================
        # 🕵️ 新增：對手動態追蹤器 (Opponent Tracker)
        # ==========================================
        self.history_window = 100 # 追蹤最近 30 局的對手行為
        self.opp_actions_history = [] # 儲存每一局對手是否有 VPIP 和 Raise
        self.current_hand_opp_vpip = False
        self.current_hand_opp_raise = False
        self.opponent_style = "Unknown"
        self.last_hand_number = 0
        
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
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        act_start_time = time.perf_counter()

        current_hand = info.get('hand_number', 1)
        hands_left = self.total_hands - current_hand + 1

        # ==========================================
        # 🕵️ 更新對手追蹤器情報
        # ==========================================
        # 1. 偵測是否為新的一局
        if current_hand != self.last_hand_number:
            if self.last_hand_number > 0:
                # 儲存上一局對手的總結表現
                self.opp_actions_history.append({
                    'vpip': self.current_hand_opp_vpip,
                    'raise': self.current_hand_opp_raise
                })
                # 保持記憶窗口大小
                if len(self.opp_actions_history) > self.history_window:
                    self.opp_actions_history.pop(0)
            
            # 結算與重置新一局的狀態
            self.last_hand_number = current_hand
            self.current_hand_opp_vpip = False
            self.current_hand_opp_raise = False
            
            # 2. 分析最近 30 局的對手風格
            if len(self.opp_actions_history) >= 10: # 收集 10 局以上才開始貼標籤
                recent_vpip_rate = sum(1 for h in self.opp_actions_history if h['vpip']) / len(self.opp_actions_history)
                recent_pfr_rate = sum(1 for h in self.opp_actions_history if h['raise']) / len(self.opp_actions_history)
                
                if recent_pfr_rate >= 0.25:
                    self.opponent_style = "Aggro" # 瘋狗：常加注
                elif recent_vpip_rate <= 0.20:
                    self.opponent_style = "Nit"   # 岩石：只玩大牌
                elif recent_vpip_rate >= 0.50 and recent_pfr_rate < 0.15:
                    self.opponent_style = "CallingStation" # 跟注站：愛看牌但不加注
                else:
                    self.opponent_style = "Normal"
        
        # 3. 即時監聽對手在【這一局】的動作 (opp_bet > 2 代表對手不僅付了盲注，還主動跟注或加注)
        my_bet        = observation["my_bet"]
        opp_bet       = observation["opp_bet"]
        if opp_bet > 2:
            self.current_hand_opp_vpip = True
        if opp_bet > my_bet and opp_bet > 4: # 代表對手主動加注
            self.current_hand_opp_raise = True

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
                    
                elif preflop_equity > 0.42:
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
        # 第五步：Post-flop (Street 1~3) 絕對勝率精算與下注邏輯
        # =========================================================================
        # 走到這裡，保證手上只有 2 張底牌，直接啟動我們完美的 O(1) 字典窮舉引擎！
        # strength = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
        global GLOBAL_EHS_TABLE
        
        HS = 0.5
        PPot = 0.0
        NPot = 0.0
        
        # 根據不同的 Street 獲取情報
        if street in (1, 2) and GLOBAL_EHS_TABLE: 
            # Flop & Turn: 查 EHS 表 (Key 是你手牌加上公牌的排序 Tuple)
            # state_key = tuple(sorted(list(my_cards) + list(community)))
            state_key = (tuple(sorted(my_cards)), tuple(sorted(community)))
            if state_key in GLOBAL_EHS_TABLE:
                HS, PPot, NPot = GLOBAL_EHS_TABLE[state_key]
            else:
                HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
        else:
            # River (Street 3) 或字典沒載入：直接查絕對勝率 (沒有潛力了)
            HS = get_exact_strength(my_cards, community, opp_discarded, my_discarded)
            
        # 計算 EHS (有效牌力)
        ehs_val = HS + (1 - HS) * PPot - HS * NPot

        opp_aggressive = (opp_bet > my_bet and opp_bet > 30)
        if self.opponent_style == "Aggro":
            adj = ehs_val + 0.05
        else:
            adj = ehs_val

        pot_odds = call_amount / (pot + call_amount + 1e-9)

        # ---------------------------------------------------------------
        # 【核心重構】絕對 EHS 跟注門檻：依 street 和底池大小設硬性下限
        # 
        # 問題根源：PlayerAgent 帶 EHS 0.67~0.71 打到大底池攤牌，
        # 而 V10 攤牌時牌力 0.79~0.84，系統性被佔便宜。
        # 解法：對「跟注大注」設硬性 EHS 門檻，低於門檻直接放棄。
        # ---------------------------------------------------------------
        def get_call_threshold(street: int, pot: int, call_amount: int) -> float:
            """
            回傳跟注所需的最低 EHS 門檻。
            street 越晚、底池越大、注碼越高 → 門檻越高。
            """
            # 基礎門檻（依 street）
            # River 沒有潛力，門檻最高；Flop 有兩條街可以追牌，門檻較低
            base = {1: 0.62, 2: 0.68, 3: 0.74}.get(street, 0.62)
            
            # 底池規模加成
            if pot > 40:
                base += 0.06
            elif pot > 25:
                base += 0.04
            elif pot > 15:
                base += 0.02
            
            # 注碼比例加成（call_amount 佔底池越大，門檻越高）
            call_ratio = call_amount / (pot + 1e-9)
            if call_ratio > 0.5:
                base += 0.04
            elif call_ratio > 0.3:
                base += 0.02
            
            return min(base, 0.88)  # 上限 0.88，避免永遠折疊

        call_threshold = get_call_threshold(street, pot, call_amount)

        # ---------------------------------------------------------------
        # 🛑 智慧型高方差防禦護盾 (Smart Shield) - 保留對手風格調整
        # ---------------------------------------------------------------
        trigger_amount = 15
        if self.opponent_style == "Nit":
            trigger_amount = 10
        elif self.opponent_style == "Aggro":
            trigger_amount = 25

        if call_amount > trigger_amount:
            risk_premium = (call_amount / 100.0) * 0.25

            if self.opponent_style == "Nit":
                safe_call_threshold = min(0.92, call_threshold + 0.05)
            elif self.opponent_style == "CallingStation":
                safe_call_threshold = min(0.82, call_threshold)
            elif self.opponent_style == "Aggro":
                safe_call_threshold = min(0.75, call_threshold - 0.05)  # 抓詐唬放寬一點
            else:
                safe_call_threshold = call_threshold

            # 脆弱度判定
            vulnerability_limit = 0.35 if self.opponent_style == "Aggro" else 0.25
            if NPot > vulnerability_limit:
                self.logger.info(f"Shield: High vulnerability (NPot={NPot:.2f}), folding.")
                return check() if can_check else fold()

            if adj < safe_call_threshold:
                self.logger.info(f"Shield: adj={adj:.2f} < threshold={safe_call_threshold:.2f} (street={street}, pot={pot})")
                return check() if can_check else fold()

        # ---------------------------------------------------------------
        # 動態加注尺度：底池大時用小尺度鎖住，底池小時用大尺度建底池
        # ---------------------------------------------------------------
        # def smart_raise_fraction(strength: float) -> float:
        #     if strength > 0.82:
        #         return 0.5 if pot > 25 else 0.8
        #     elif strength > 0.72:
        #         return 0.35 if pot > 20 else 0.45
        #     else:
        #         return 0.3
        def smart_raise_fraction(strength: float, pot: int) -> float:
            # 1. 天牌區間 (0.82 ~ 1.0)
            # 平滑邏輯：從 0.5 起跳，最高可以加注到 1.0 (全下或大注)
            if strength > 0.82:
                # 當 strength 從 0.82 爬升到 1.0 時，額外增加 0 ~ 0.5 的注碼
                base = 0.5 if pot > 25 else 0.6
                bonus = (strength - 0.82) * (1.0 / 0.18) * 0.4  # 最大再加 40%
                return min(1.0, base + bonus)

            # 2. 強牌區間 (0.72 ~ 0.82)
            # 平滑邏輯：從 0.25 起跳，隨著牌力逼近 0.82，加注比例平滑上升至 0.5
            elif strength > 0.72:
                base = 0.25
                # 當 strength 從 0.72 爬升到 0.82 時，額外增加 0 ~ 0.25 的注碼
                bonus = (strength - 0.72) * (1.0 / 0.10) * 0.25
                return min(0.6, base + bonus)
                
            # 3. 中等牌力或半詐唬
            else:
                return 0.3


        # Anti-bully check-raise（對付瘋狗，需要真正強牌才回打）
        if self.opponent_style == "Aggro":
            if adj > 0.72 and can_raise:
                return make_raise(smart_raise_fraction(adj, pot))

        # 1. 天牌
        if adj > 0.82 or (HS > 0.80 and NPot < 0.15):
            if can_raise:
                return make_raise(smart_raise_fraction(adj, pot))
            return call() if can_call else check()

        # 2. 強牌主動加注區間（0.78+）
        # 【核心修改】把主動加注門檻拉到 0.78，對齊 V10 贏牌時的 EHS (0.82~0.84)
        # 0.68~0.78 的牌改為「底池控制」：只 check 或跟小注，不主動建大底池
        if adj > 0.78:
            if can_raise:
                return make_raise(smart_raise_fraction(adj, pot))
            return call() if can_call else check()

        # 3. 中上牌力 (0.60~0.78)：底池控制區
        # 不主動加注，只 check 或跟合理的注。這樣進入攤牌時對手也不會有大底池優勢。
        if adj > 0.60:
            # check-raise 機會：對手下注且牌力夠（0.72+），才回打
            if can_raise and call_amount > 0 and adj > 0.72:
                return make_raise(smart_raise_fraction(adj, pot))

            # 跟注：用硬性門檻
            if can_call:
                effective_threshold = call_threshold - 0.05
                if adj > max(pot_odds + 0.10, effective_threshold):
                    return call()
            if can_check: return check()
            return fold()

        # 4. 半詐唬聽牌（低 HS 但有潛力）
        # 放寬門檻：PPot 0.35→0.28，讓更多聽牌可以施壓
        if HS < 0.55 and PPot > 0.28:
            if can_raise:
                if self.opponent_style in ["CallingStation", "Nit"]:
                    if can_call and call_amount <= max(10, int(pot * 0.4)):
                        return call()
                    return check() if can_check else fold()
                elif random.random() < 0.70:  # 0.65 → 0.70
                    return make_raise(smart_raise_fraction(adj, pot))
            if can_call and call_amount <= max(8, int(pot * 0.35)):
                return call()
            if can_check: return check()
            return fold()

        # 5. 邊緣牌力：只在底池小時便宜看牌
        if adj > 0.42:
            if can_check: return check()
            max_call = max(2, int(pot * (0.15 if pot > 20 else 0.22)))
            if can_call and call_amount <= max_call:
                return call()
            return fold()
        
        # 6. 垃圾牌：詐唬邏輯（依 street 差異化）
        # Flop 詐唬最有價值（還有兩條街施壓），River 純詐唬風險最高
        if self.opponent_style == "Aggro":
            base_raise_bluff = 0.25
        elif street == 1:   # Flop
            base_raise_bluff = 0.18
        elif street == 2:   # Turn
            base_raise_bluff = 0.13
        else:               # River
            base_raise_bluff = 0.10
        base_call_bluff = 0.10
        
        if call_amount == 0 or call_amount <= max(2, int(pot * 0.1)):
            base_raise_bluff += 0.08
        elif call_amount > int(pot * 0.4):
            base_raise_bluff = 0.0
            base_call_bluff  = 0.0
            
        if street == 3:
            base_raise_bluff *= 0.5
            base_call_bluff  *= 0.1
        
        jitter = random.uniform(-0.01, 0.01)
        final_raise_bluff = max(0.0, base_raise_bluff + jitter)
        final_call_bluff  = max(0.0, base_call_bluff + jitter)
        
        bluff_roll = random.random()
        
        if bluff_roll < final_raise_bluff and can_raise:
            return make_raise(random.uniform(0.25, 0.45))
            
        elif bluff_roll < (final_raise_bluff + final_call_bluff) and can_call and call_amount <= max(4, int(pot * 0.3)):
            return call()
            
        if can_check: return check()
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