import random
import itertools
from collections import Counter
from functools import lru_cache  # <--- 引入 Python 的内存缓存神器

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval
import time

# 引入官方动作和卡牌翻译官
ActionType = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

# 初始化官方牌力评估神器
evaluator = WrappedEval()

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
    """性能优化1：缓存7张牌的算牌结果，遇到同样的牌面直接从内存拿结果，不重算"""
    best_score = float('inf')
    for combo in itertools.combinations(all_cards_tuple, 5):
        score = evaluator.evaluate(list(combo), [])
        if score < best_score:
            best_score = score
    return best_score

def evaluate_best_hand(hole_str: list, board_str: list) -> int:
    # 转成 tuple 并排序，这是使用缓存的前提
    all_cards_tuple = tuple(sorted(hole_str + board_str))
    return cached_evaluate_best_hand(all_cards_tuple)

# ---------------------------------------------------------------------------
# Monte Carlo simulation (Optimized & Cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def cached_monte_carlo(my_cards_tuple, community_tuple, opp_discarded_tuple, my_discarded_tuple, n_sim: int) -> float:
    """性能优化2：缓存整场模拟结果，遇到同样的底牌+公共牌（比如加注大战），直接返回胜率"""
    known = set(my_cards_tuple) | set(community_tuple) | set(opp_discarded_tuple) | set(my_discarded_tuple)
    deck_ints = [c for c in range(27) if c not in known]

    board_needed = 5 - len(community_tuple)
    wins = ties = total = 0

    # 【性能优化3】：把繁重的转换工作移出循环！在这只翻译1次，而不是循环里翻译200次！
    deck_strs = [int_to_card(c) for c in deck_ints]
    my_cards_str = [int_to_card(c) for c in my_cards_tuple]
    comm_str = [int_to_card(c) for c in community_tuple]

    opp_hole_count = len(my_cards_tuple) 
    need = board_needed + opp_hole_count

    for _ in range(n_sim):
        if len(deck_strs) < need:
            break
            
        # 直接从已经变成字符串的牌库里抽牌，速度飙升
        sample_strs = random.sample(deck_strs, need)
        sim_board_str = comm_str + sample_strs[:board_needed]
        opp_hole_str  = sample_strs[board_needed:]

        my_score  = evaluate_best_hand(my_cards_str, sim_board_str)
        opp_score = evaluate_best_hand(opp_hole_str, sim_board_str)

        if my_score < opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total

def monte_carlo_strength(my_cards: list, community: list, opp_discarded: list, my_discarded: list, n_sim: int = 200) -> float:
    # 列表是不能被缓存的，必须转成不可变的 Tuple
    return cached_monte_carlo(tuple(my_cards), tuple(community), tuple(opp_discarded), tuple(my_discarded), n_sim)


def choose_discard(
    hole: list,
    community: list,
    opp_discarded: list,
    my_discarded: list,
    n_sim: int = 50 
) -> tuple:
    best_idx = (0, 1)
    best_str = -1.0
    for i, j in itertools.combinations(range(5), 2):
        kept = [hole[i], hole[j]]
        s = monte_carlo_strength(kept, community, opp_discarded, my_discarded, n_sim)
        if s > best_str:
            best_str = s
            best_idx = (i, j)
    return best_idx, best_str

# ---------------------------------------------------------------------------
# Player Agent
# ---------------------------------------------------------------------------
class V10(Agent):

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = ActionType
        # --- 新增：全域資源與狀態追蹤 ---
        self.start_time = time.perf_counter()
        # 設定安全時間閾值 (Phase 3 限制為 1500 秒，預留 50 秒作為緩衝)
        self.time_limit = 450.0 
        self.total_hands = 1000
        self.net_chips = 0.0  # 追蹤我方累積淨收益
        self.is_guaranteed_win = False
        self.my_total_think_time = 0.0

    def __name__(self):
        return "V10"

    def act(self, observation, reward, terminated, truncated, info):
        act_start_time = time.perf_counter()
        # self.logger.info(f"Info Keys: {info.keys()}, Reward: {reward}")
        # self.logger.info(f"Hand {info.get('hand_number', '?')} street {observation['street']}")

        current_hand = info.get('hand_number', 1)
        hands_left =    self.total_hands - current_hand + 1


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

        # ── Helpers ───────────────────────────────────────────────────
        def record_and_return(action_tuple):
            # 結算本次 act() 真正花費的時間，並存入總思考時間
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
        # 如果還沒鎖定，檢查是否達標
        if not self.is_guaranteed_win:
            # max_possible_loss = hands_left * 2
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            if self.net_chips > max_possible_loss:
                self.is_guaranteed_win = True  # 狀態切換：永久進入必勝模式
                self.logger.info(f"狀態切換：必勝鎖定！淨賺 {self.net_chips} > 極限損失 {max_possible_loss}。永久進入龜縮模式。")

        # 只要處於必勝狀態，無腦執行 O(1) 逃脫邏輯
        if self.is_guaranteed_win:
            # 強制換牌階段 (Street 1) 必須選兩張牌保留 [cite: 36, 449]
            if valid_actions[ActionType.DISCARD.value]:
                return do_discard(0, 1)
            if valid_actions[ActionType.FOLD.value]:
                return fold()
            return check()

        # =========================================================================
        # 第二步：加權預算池與動態算力分配 (Weighted Time Pool)
        # =========================================================================
        elapsed_time = time.perf_counter() - self.start_time
        # time_remaining = max(0.1, self.time_limit - elapsed_time)
        time_remaining = max(0.1, self.time_limit - self.my_total_think_time)

        # 1. 設定我們「理想中」每手牌的耗時目標 (前期給極大寬容度，後期壓縮)
        early_cost_target = 0.85  # 前 400 手，每把允許吃 0.85 秒
        late_cost_target = 0.15   # 後 600 手，每把只給 0.15 秒 (因為有很多局會直接 Fold，0.15 很夠)

        # 2. 計算剩餘回合數 (嚴格區分前期與後期)
        if current_hand <= 400:
            early_hands_left = 400 - current_hand + 1
            late_hands_left = self.total_hands - 400
            
            # 前期：較高的基礎模擬次數
            base_n_sim = {0: 100, 1: 150, 2: 200, 3: 250}.get(street, 100)
        else:
            early_hands_left = 0
            late_hands_left = self.total_hands - current_hand + 1
            
            # 中後期：保守的模擬次數
            base_n_sim = {0: 50, 1: 100, 2: 150, 3: 200}.get(street, 50)

        # 3. 結算：如果照這個奢侈的計畫打到最後，還需要多少秒？
        total_needed_budget = (early_hands_left * early_cost_target) + (late_hands_left * late_cost_target)

        # 4. 算力倍率：我擁有的時間 / 我需要的時間
        sim_multiplier = time_remaining / max(1.0, total_needed_budget)

        # 5. 安全鉗制與套用
        if sim_multiplier < 0.9:
            # self.logger.warning(f"⚠️ 總預算落後！啟動降速，multiplier: {sim_multiplier:.2f}")
            self.logger.warning(f"Time budget fall behind, reduce speed multiplier: {sim_multiplier:.2f}")
        
        # 防止過度膨脹或過度壓縮 (最高 2.0 倍，最低 0.2 倍)
        sim_multiplier = max(0.2, min(2.0, sim_multiplier))

        n_sim = int(base_n_sim * sim_multiplier)
        n_sim = max(10, n_sim) # 底線防禦
        
        self.logger.info(f"Hand {current_hand} | Street {street} | Time left: {time_remaining:.1f}s | multiplier: {sim_multiplier:.2f} | n_sim: {n_sim}")

        # =========================================================================
        # 第三步：處理強制換牌 (現在可以吃到動態 n_sim 了)
        # =========================================================================
        if valid_actions[ActionType.DISCARD.value]:
            if len(my_cards) == 5:
                # 這裡補上 n_sim，讓換牌決策也能根據剩餘時間縮放算力
                best_idx, _ = choose_discard(my_cards, community, opp_discarded, my_discarded, n_sim=n_sim)
                i, j = best_idx
            else:
                i, j = 0, 1
            return do_discard(i, j)
        
        # =========================================================================
        # 第四步：常規下注與詐唬邏輯
        # =========================================================================
        # ── Safety check ──────────────────────────────────────────────
        if len(my_cards) < 2:
            return check() if can_check else fold()
        # ── Estimate hand strength ─────────────────────────────────────
        if len(my_cards) == 5:
            _, strength = choose_discard(my_cards, community, opp_discarded, my_discarded, n_sim = n_sim)
        else:
            strength = monte_carlo_strength(my_cards, community, opp_discarded, my_discarded, n_sim)

        # 微调：面对强力加注时稍微扣点胜率预估
        opp_aggressive = (opp_bet > my_bet and opp_bet > 4)
        adj = strength - (0.05 if opp_aggressive else 0.0)

        pot_odds = call_amount / (pot + call_amount + 1e-9)

        # 🛑 新增：高方差防禦護盾 (Anti-Variance Shield)
        if call_amount > 35:
            # 根據實際威脅程度 (佔最大下注的比例) 計算風險溢價，最高要求額外 15% 勝率
            risk_premium = (call_amount / 100.0) * 0.15 
            
            # 動態安全線：底池賠率 + 風險溢價
            safe_call_threshold = pot_odds + risk_premium
            
            # 絕對底線防禦：設定在 0.65 到 0.70 之間比較適合 27 張牌的生態
            # 確保不會要求不切實際的高勝率
            safe_call_threshold = min(0.70, safe_call_threshold)
            
            if adj < safe_call_threshold:
                # self.logger.info(f"🛡️ 觸發防護盾：拒絕極端下注！要求跟注 {call_amount}，牌力 {adj:.2f} < 安全線 {safe_call_threshold:.2f}")
                self.logger.info(f"Shield: No extreme bid, call amount: {call_amount}, ajd: {adj:.2f} < safty_threshold: {safe_call_threshold:.2f}")
                return check() if can_check else fold()

        # ── Betting decisions (with bluffing logic)────────────────────────────────
        if adj > 0.75:
            if can_raise:
                frac = min(1.0, 0.5 + (adj - 0.75) * 2.0)
                return make_raise(frac)
            if can_call:  return call()
            return check()

        if adj > 0.60:
            if can_raise and adj > 0.68:
                return make_raise(0.25)
            if can_call and adj > pot_odds + 0.05:
                return call()
            if can_check: return check()
            if can_call:  return call()
            return fold()

        if adj > 0.42:
            if can_check: return check()
            if can_call and call_amount <= max(2, int(pot * 0.25)):
                return call()
            return fold()
        
        # 4. 烂牌 (< 42%)：启动动态诈唬机制 (Dynamic Bluffing)
        base_raise_bluff = 0.05  
        base_call_bluff  = 0.05  
        
        if call_amount == 0 or call_amount <= max(2, int(pot * 0.1)):
            base_raise_bluff += 0.12  
        elif call_amount > int(pot * 0.5):
            base_raise_bluff = 0.0    
            base_call_bluff  = 0.0
            
        if street == 3: 
            base_raise_bluff *= 0.2
            base_call_bluff  *= 0.2
            
        jitter = random.uniform(-0.02, 0.03)
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
        """
        環境廣播狀態與結算的隱藏函數。
        當手牌結束 (terminated = True) 時，這裡的 reward 就是你這把牌真正的淨收益！
        """
        # 1. 攔截真正的結算籌碼
        if reward is not None and reward != 0.0:
            self.net_chips += reward
            self.logger.info(f"Getting current game reward : {reward}, Overall reward: {self.net_chips}")

        # 2. 可選：當手牌結束時，印出目前的進度，方便你監控必勝鎖定是否快觸發了
        if terminated:
            current_hand = info.get('hand_number', 1)
            hands_left = self.total_hands - current_hand
            # max_possible_loss = hands_left * 2
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            self.logger.info(f"🏁 Hand {current_hand} finished | Total earnings: {self.net_chips} | Distance to locking win (Max possible loss): {max_possible_loss}")
            
            # 你也可以保留對手的 showdown 資訊方便覆盤
            # if "player_0_cards" in info:
                # self.logger.debug(f"攤牌: {info['player_0_cards']} vs {info['player_1_cards']} ボード {info['community_cards']}")