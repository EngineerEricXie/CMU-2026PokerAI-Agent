import random
import itertools
from collections import Counter
from functools import lru_cache  # <--- 引入 Python 的内存缓存神器

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval

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
def cached_monte_carlo(my_cards_tuple, community_tuple, opp_discarded_tuple, n_sim: int) -> float:
    """性能优化2：缓存整场模拟结果，遇到同样的底牌+公共牌（比如加注大战），直接返回胜率"""
    known = set(my_cards_tuple) | set(community_tuple) | set(opp_discarded_tuple)
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

def monte_carlo_strength(my_cards: list, community: list, opp_discarded: list, n_sim: int = 200) -> float:
    # 列表是不能被缓存的，必须转成不可变的 Tuple
    return cached_monte_carlo(tuple(my_cards), tuple(community), tuple(opp_discarded), n_sim)


def choose_discard(
    hole: list,
    community: list,
    opp_discarded: list,
    n_sim: int = 50 
) -> tuple:
    best_idx = (0, 1)
    best_str = -1.0
    for i, j in itertools.combinations(range(5), 2):
        kept = [hole[i], hole[j]]
        s = monte_carlo_strength(kept, community, opp_discarded, n_sim)
        if s > best_str:
            best_str = s
            best_idx = (i, j)
    return best_idx, best_str

# ---------------------------------------------------------------------------
# Player Agent
# ---------------------------------------------------------------------------
class PlayerAgent(Agent):

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = ActionType

    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        # self.logger.info(f"Hand {info.get('hand_number', '?')} street {observation['street']}")

        valid_actions = observation["valid_actions"]
        my_cards      = valid_cards(observation["my_cards"])
        community     = valid_cards(observation["community_cards"])
        opp_discarded = valid_cards(observation["opp_discarded_cards"])
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
        def make_raise(fraction: float):
            amt = int(min_raise + (max_raise - min_raise) * fraction)
            amt = max(min_raise, min(max_raise, amt))
            return ActionType.RAISE.value, amt, 0, 0

        def fold():  return ActionType.FOLD.value,  0, 0, 0
        def call():  return ActionType.CALL.value,  0, 0, 0
        def check(): return ActionType.CHECK.value, 0, 0, 0

        # ── Discard round (mandatory on flop) ─────────────────────────
        if valid_actions[ActionType.DISCARD.value]:
            if len(my_cards) == 5:
                best_idx, _ = choose_discard(my_cards, community, opp_discarded)
                i, j = best_idx
            else:
                i, j = 0, 1
            return ActionType.DISCARD.value, 0, i, j

        # ── Safety check ──────────────────────────────────────────────
        if len(my_cards) < 2:
            return check() if can_check else fold()

        # ── Estimate hand strength ─────────────────────────────────────
        # 因为现在速度极快，你甚至可以把这里的模拟次数往上调了！我们暂且保留原参数，保证100%不超时。
        n_sim = {0: 50, 1: 100, 2: 150, 3: 200}.get(street, 50) 
        
        if len(my_cards) == 5:
            _, strength = choose_discard(my_cards, community, opp_discarded, n_sim = n_sim)
        else:
            strength = monte_carlo_strength(my_cards, community, opp_discarded, n_sim)

        # 微调：面对强力加注时稍微扣点胜率预估
        opp_aggressive = (opp_bet > my_bet and opp_bet > 4)
        adj = strength - (0.05 if opp_aggressive else 0.0)

        pot_odds = call_amount / (pot + call_amount + 1e-9)

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