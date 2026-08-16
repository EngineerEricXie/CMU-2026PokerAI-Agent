import random
import itertools
from collections import Counter
from functools import lru_cache  # <--- 引入 Python 的内存缓存神器

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval
import time
import os
import json

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
class V11(Agent):

    def __init__(self, stream: bool = True, player_id=None):
        super().__init__(stream, player_id=player_id)
        self.action_types = ActionType
        # --- 新增：全域資源與狀態追蹤 ---
        self.start_time = time.perf_counter()
        # 設定安全時間閾值 (Phase 3 限制為 1500 秒，預留 50 秒作為緩衝)
        self.time_limit = 450.0 
        self.total_hands = 1000
        self.net_chips = 0.0  # 追蹤我方累積淨收益
        self.is_guaranteed_win = False
        self.my_total_think_time = 0.0

        self.dynamic_thresholds = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # for mc in [10, 30, 50, 100, 150, 200, 250]:
        for mc in [50, 100, 150, 200]:
            filename = os.path.join(base_dir, f"thresholds_mc{mc}.json")
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        mc_key = int(list(data.keys())[0])
                        self.dynamic_thresholds[mc_key] = {int(k): v for k, v in data[str(mc_key)].items()}
                    # 可以取消註解下面這行來確認是否成功讀取
                    self.logger.info(f"Read successfully: {filename}")
                except Exception as e:
                    self.logger.warning(f"Reading {filename} error: {e}")
        # 如果因為任何原因 (例如忘記上傳 JSON 到比賽伺服器) 導致沒讀到任何檔案
        if not self.dynamic_thresholds:
            self.logger.warning("Cannot find json file, use hard coded value。")
            # 放一組偏保守的預設數字 (大約是 mc_sims=150 的水準)
            self.dynamic_thresholds[150] = {
                0: {"raise": 0.690, "call": 0.640, "marginal": 0.613, "PR_80": 0.660, "PR_60": 0.625, "PR_50": 0.613},
                1: {"raise": 0.877, "call": 0.760, "marginal": 0.680},
                2: {"raise": 0.940, "call": 0.800, "marginal": 0.687},
                3: {"raise": 0.990, "call": 0.880, "marginal": 0.732},
            }

    def __name__(self):
        return "V11"

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
                # return ActionType.DISCARD.value, 0, 0, 1
                return do_discard(0, 1)
            if valid_actions[ActionType.FOLD.value]:
                return fold()
            return check()

        # =========================================================================
        # 第二步：加權預算池與動態算力分配 (Weighted Time Pool)
        # =========================================================================
        elapsed_time = time.perf_counter() - self.start_time
        # time_remaining = max(0.1, self.time_limit - elapsed_time)
        # 🏆 修正：改用我們自己的專屬碼表來計算剩餘時間
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
            self.logger.warning(f"Time budget fall behind, reduce speed multiplier: {sim_multiplier:.2f}")
        
        # 防止過度膨脹或過度壓縮 (最高 2.0 倍，最低 0.2 倍)
        sim_multiplier = max(0.2, min(2.0, sim_multiplier))

        n_sim = int(base_n_sim * sim_multiplier)
        n_sim = max(10, n_sim) # 底線防禦
        # n_sim = int(round(n_sim / 10.0) * 10)
        
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
        # 第四步：常規下注與詐唬邏輯 (搭載蒙地卡羅分佈測量數據)
        # =========================================================================
        # ── Safety check ──────────────────────────────────────────────
        if len(my_cards) < 2:
            return check() if can_check else fold()
            
        # ── Estimate hand strength ─────────────────────────────────────
        if len(my_cards) == 5:
            _, strength = choose_discard(my_cards, community, opp_discarded, my_discarded, n_sim=n_sim)
        else:
            strength = monte_carlo_strength(my_cards, community, opp_discarded, my_discarded, n_sim)

        opp_aggressive = (opp_bet > my_bet and opp_bet > 4)
        adj = strength - (0.05 if opp_aggressive else 0.0)
        pot_odds = call_amount / (pot + call_amount + 1e-9)

        # 🏆 根據實測數據建立的動態閾值表 (Thresholds)
        # ! Old 1
        # 格式: {street: {"raise": PR90, "call": PR70, "marginal": PR50}}
        # dynamic_thresholds = {
        #     0: {"raise": 0.690, "call": 0.640, "marginal": 0.613},
        #     1: {"raise": 0.877, "call": 0.760, "marginal": 0.680},
        #     2: {"raise": 0.940, "call": 0.800, "marginal": 0.687},
        #     3: {"raise": 0.990, "call": 0.880, "marginal": 0.732}, # PR90 稍微下調到 0.99 避免太龜
        # }
        # ! Old 2
        # if n_sim >= 200:
        #     sim_level = 200
        # elif n_sim >= 150:
        #     sim_level = 150
        # elif n_sim >= 100:
        #     sim_level = 100
        # else:
        #     sim_level = 30
        
        # t_raise = self.dynamic_thresholds[sim_level][street]["raise"]
        # t_call  = self.dynamic_thresholds[sim_level][street]["call"]
        # t_marg  = self.dynamic_thresholds[sim_level][street]["marginal"]
        # ! Old 3
        # =========================================================================
        # 動態閾值：線性插值 (Linear Interpolation) 處理算力波動
        # =========================================================================
        # if street == 0:
        #     k_raise = "PR_80"  # Pre-flop 放寬 Raise 標準 (原本是 PR90)
        #     k_call  = "PR_60"  # Pre-flop 放寬 Call 標準 (原本是 PR70)
        #     k_marg  = "PR_50"  # 邊緣牌抓 PR50，稍後手動再降一點
        # else:
        #     k_raise = "raise"
        #     k_call  = "call"
        #     k_marg  = "marginal"
        # ! New
        # =========================================================================
        # 動態閾值：錦標賽通殺策略 (Tournament Baseline - LAG/TAG 混合)
        # =========================================================================
        if street == 0:
            # Pre-flop: 極度寬鬆！捍衛盲注，多看翻牌
            k_raise = "PR_70"  # 前 30% 就主動加注，壓制對手
            k_call  = "PR_50"  # 前 50% 都跟注 (甚至 PR_30 也可以)
            k_marg  = "PR_50"  # 稍後用 offset 降到 PR_30
        elif street == 1:
            # Flop (換牌後): 牌力初步成型，開始過濾爛牌
            k_raise = "PR_80" 
            k_call  = "PR_60"  
            k_marg  = "PR_50"
        elif street == 2:
            # Turn: 轉牌圈，牌力逐漸明朗，抓緊價值
            k_raise = "PR_90"  
            k_call  = "PR_60"
            k_marg  = "PR_50"
        else: 
            # River: 對手圖窮匕見，拿前 20% (PR_80) 的牌狠狠榨取他們
            k_raise = "PR_90"  
            k_call  = "PR_60"
            k_marg  = "PR_50"
            
        k_nuts = "PR_95" # 🏆 新增：用來懲罰瘋魚的極限堅果牌 Key

        # 1. 取得你目前載入的所有算力錨點並排序 (例如: [30, 100, 150, 200])
        anchor_levels = sorted(list(self.dynamic_thresholds.keys()))
        
        # 2. 邊界鉗制：如果算力超出我們測量的極限，直接使用極值
        if n_sim <= anchor_levels[0]:
            t_raise = self.dynamic_thresholds[anchor_levels[0]][street][k_raise]
            t_call  = self.dynamic_thresholds[anchor_levels[0]][street][k_call]
            t_marg  = self.dynamic_thresholds[anchor_levels[0]][street][k_marg]
            t_nuts  = self.dynamic_thresholds[anchor_levels[0]][street].get(k_nuts, 0.95)
        elif n_sim >= anchor_levels[-1]:
            t_raise = self.dynamic_thresholds[anchor_levels[-1]][street][k_raise]
            t_call  = self.dynamic_thresholds[anchor_levels[-1]][street][k_call]
            t_marg  = self.dynamic_thresholds[anchor_levels[-1]][street][k_marg]
            t_nuts  = self.dynamic_thresholds[anchor_levels[-1]][street].get(k_nuts, 0.95)
        else:
            # 3. 尋找 n_sim 落在哪兩個錨點之間 (x0 和 x1)
            for i in range(len(anchor_levels) - 1):
                x0 = anchor_levels[i]
                x1 = anchor_levels[i+1]
                
                if x0 <= n_sim <= x1:
                    # 計算插值比例 (0.0 ~ 1.0)
                    ratio = (n_sim - x0) / (x1 - x0)
                    
                    # 抓出 x0 和 x1 對應的 y 值
                    y0_raise = self.dynamic_thresholds[x0][street][k_raise]
                    y1_raise = self.dynamic_thresholds[x1][street][k_raise]
                    
                    y0_call  = self.dynamic_thresholds[x0][street][k_call]
                    y1_call  = self.dynamic_thresholds[x1][street][k_call]
                    
                    y0_marg  = self.dynamic_thresholds[x0][street][k_marg]
                    y1_marg  = self.dynamic_thresholds[x1][street][k_marg]
                    
                    # 處理防呆：萬一舊的 JSON 沒存到 PR_95，預設給 0.95
                    y0_nuts  = self.dynamic_thresholds[x0][street].get(k_nuts, 0.95)
                    y1_nuts  = self.dynamic_thresholds[x1][street].get(k_nuts, 0.95)
                    
                    # 執行數學線性插值公式
                    t_raise = y0_raise + ratio * (y1_raise - y0_raise)
                    t_call  = y0_call  + ratio * (y1_call  - y0_call)
                    t_marg  = y0_marg  + ratio * (y1_marg  - y0_marg)
                    t_nuts  = y0_nuts  + ratio * (y1_nuts  - y0_nuts)
                    break
        
        # 🛡️ 微調：讓 Street 0 的邊緣及格線再低一點，避免一開始瘋狂棄牌
        if street == 0:
            t_marg -= 0.08
        
        # 🛑 高方差防禦護盾 (Anti-Variance Shield)
        if call_amount > 35:
            risk_premium = (call_amount / 100.0) * 0.15 
            safe_call_threshold = pot_odds + risk_premium
            
            # 使用當前 Street 的 t_call 作為絕對底線，避免用爛牌接 All-in
            safe_call_threshold = min(t_call, safe_call_threshold)
            
            if adj < safe_call_threshold:
                self.logger.info(f"🛡️ 觸發防護盾：要求跟注 {call_amount}, 牌力 {adj:.2f} < 安全線 {safe_call_threshold:.2f}")
                return check() if can_check else fold()

        # ── Betting decisions (智能匹配當前 Street 的強度) ────────────────
        # 1. 超強牌 (前 20% ~ 30%)
        if adj >= t_raise:
            if can_raise:
                # 💥 瘋魚懲罰機制：如果對手下注極大，且我們牌力大於插值算出來的極限堅果線 (t_nuts)
                if opp_aggressive and adj >= t_nuts:
                    return make_raise(1.0) # 直接 Max Raise / All-in
                
                # 根據超過閾值的程度決定加注大小 (越接近 PR95 加注越重)
                frac = min(1.0, 0.5 + (adj - t_raise) * 2.5)
                return make_raise(frac)
            if can_call:  return call()
            return check()

        # 2. 強牌 (前 30%)
        if adj >= t_call:
            if can_raise and adj > (t_raise + t_call) / 2.0:
                return make_raise(0.25) # 接近超強牌時可以小建樹
            if can_call and adj > pot_odds + 0.05:
                return call()
            if can_check: return check()
            if can_call:  return call()
            return fold()

        # 3. 中庸牌 (前 50%)
        if adj >= t_marg:
            if can_check: return check()
            # 只有在代價很小的時候才跟注看牌
            if can_call and call_amount <= max(2, int(pot * 0.25)):
                return call()
            return fold()
        
        # 4. 爛牌 (< PR50)：啟動動態詐唬機制 (Dynamic Bluffing)
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
