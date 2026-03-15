import random
import itertools
from collections import Counter

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
# Hand evaluator (Using Official Engine)
# ---------------------------------------------------------------------------
def evaluate_best_hand(hole_str: list, board_str: list) -> int:
    """
    把手牌和公共牌混在一起，利用官方引擎找出威力最大的 5 张牌组合。
    返回官方的 rank 分数（分数越低，牌力越强）。
    """
    all_cards = hole_str + board_str
    best_score = float('inf')
    
    # 官方引擎每次必须恰好接收 5 张牌，所以我们穷举出所有的 5 张牌组合喂给它
    for combo in itertools.combinations(all_cards, 5):
        score = evaluator.evaluate(list(combo), [])
        if score < best_score:
            best_score = score
            
    return best_score

# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------
def monte_carlo_strength(
    my_cards: list,
    community: list,
    opp_discarded: list,
    n_sim: int = 200
) -> float:
    known = set(my_cards) | set(community) | set(opp_discarded)
    deck = [c for c in range(27) if c not in known]

    board_needed = 5 - len(community)
    wins = ties = total = 0

    for _ in range(n_sim):
        # 动态获取对手的底牌数量（保证永远是绝对公平的 2打2）
        # need = board_needed + 2
        opp_hole_count = len(my_cards) 
        need = board_needed + opp_hole_count
        if len(deck) < need:
            break
        sample = random.sample(deck, need)
        sim_board = community + sample[:board_needed]
        opp_hole  = sample[board_needed:]

        # 把整数 [0, 26] 翻译成引擎认识的字符串 ['2d', 'As']
        my_cards_str  = [int_to_card(c) for c in my_cards]
        sim_board_str = [int_to_card(c) for c in sim_board]
        opp_hole_str  = [int_to_card(c) for c in opp_hole]

        # 丢给刚才写好的进气阀，算出双方的最高战力
        my_score  = evaluate_best_hand(my_cards_str, sim_board_str)
        opp_score = evaluate_best_hand(opp_hole_str, sim_board_str)

        # 分数越低，牌力越强
        if my_score < opp_score:
            wins += 1
        elif my_score == opp_score:
            ties += 1
        total += 1

    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def choose_discard(
    hole: list,
    community: list,
    opp_discarded: list,
    n_sim: int = 50 # decreased for faster decision-making during discard round
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
        self.logger.info(
            f"Hand {info.get('hand_number', '?')} street {observation['street']}"
        )

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
                # 留出计算时间，这里的 n_sim 在 choose_discard 里默认设为 50 了
                best_idx, _ = choose_discard(my_cards, community, opp_discarded)
                i, j = best_idx
            else:
                i, j = 0, 1
            return ActionType.DISCARD.value, 0, i, j

        # ── Safety check ──────────────────────────────────────────────
        if len(my_cards) < 2:
            return check() if can_check else fold()

        # ── Estimate hand strength ─────────────────────────────────────
        # 略微调低了初始阶段的模拟次数以保证不超时
        # n_sim = {0: 100, 1: 150, 2: 200, 3: 250}.get(street, 100)
        # strength = monte_carlo_strength(my_cards, community, opp_discarded, n_sim)
        n_sim = {0: 50, 1: 100, 2: 150, 3: 200}.get(street, 50) # decreased for faster decision-making in early streets
        if len(my_cards) == 5:
            # 【核心修复】：翻牌前 (Street 0)，直接获取最佳的 2 张牌的真实胜率，拒绝虚假自信！
            _, strength = choose_discard(my_cards, community, opp_discarded, n_sim = n_sim)
        else:
            strength = monte_carlo_strength(my_cards, community, opp_discarded, n_sim)

        # 微调：面对强力加注时稍微扣点胜率预估
        opp_aggressive = (opp_bet > my_bet and opp_bet > 4)
        adj = strength - (0.05 if opp_aggressive else 0.0)

        pot_odds = call_amount / (pot + call_amount + 1e-9)

        self.logger.info(
            f"  strength={strength:.3f} adj={adj:.3f} pot_odds={pot_odds:.3f} "
            f"call={call_amount} pot={pot}"
        )

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
        
        # --- 计算动态诈唬概率 ---
        base_raise_bluff = 0.05  # 基础加注诈唬率降到 5%
        base_call_bluff  = 0.05  # 基础跟注诈唬率
        
        # 变量1：看对手脸色 (如果需要跟注的钱很少，或者对手过牌了，说明对手很软弱)
        if call_amount == 0 or call_amount <= max(2, int(pot * 0.1)):
            base_raise_bluff += 0.12  # 对手越弱，我越嚣张！诈唬加注率飙升到 17%
        elif call_amount > int(pot * 0.5):
            base_raise_bluff = 0.0    # 对手下重注，绝对不加注送死
            base_call_bluff  = 0.0
            
        # 变量2：看比赛阶段 (越往后越收敛)
        if street == 3: # 到了河牌圈，坚决少骗人
            base_raise_bluff *= 0.2
            base_call_bluff  *= 0.2
            
        # 变量3：加入随机“情绪”扰动 (让概率曲线产生噪音，防止被 RL 拟合)
        jitter = random.uniform(-0.02, 0.03)
        final_raise_bluff = max(0.0, base_raise_bluff + jitter)
        final_call_bluff  = max(0.0, base_call_bluff + jitter)
        
        # --- 开始掷骰子 ---
        bluff_roll = random.random()
        
        if bluff_roll < final_raise_bluff and can_raise:
            self.logger.info(f"  *** DYNAMIC BLUFF RAISE! (Rate: {final_raise_bluff:.2f}) ***")
            # 诈唬金额也随机化：下注底池的 25% 到 45% 之间
            bluff_fraction = random.uniform(0.25, 0.45)
            return make_raise(bluff_fraction)
            
        elif bluff_roll < (final_raise_bluff + final_call_bluff) and can_call and call_amount <= max(4, int(pot * 0.3)):
            self.logger.info(f"  *** DYNAMIC BLUFF CALL! (Rate: {final_call_bluff:.2f}) ***")
            return call()
            
        # 剩下的情况：老老实实认怂
        if can_check: return check()
        return fold()