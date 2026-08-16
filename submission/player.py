import itertools
import random
import time
from functools import lru_cache

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval

ActionType = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

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
    """Cache repeated hand evaluations for Monte Carlo simulations."""
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
# Monte Carlo simulation (Optimized & Cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2048)
def cached_monte_carlo(my_cards_tuple, community_tuple, opp_discarded_tuple, my_discarded_tuple, n_sim: int) -> float:
    """Estimate showdown equity and cache repeated public-card states."""
    known = set(my_cards_tuple) | set(community_tuple) | set(opp_discarded_tuple) | set(my_discarded_tuple)
    deck_ints = [c for c in range(27) if c not in known]

    board_needed = 5 - len(community_tuple)
    wins = ties = total = 0

    deck_strs = [int_to_card(c) for c in deck_ints]
    my_cards_str = [int_to_card(c) for c in my_cards_tuple]
    comm_str = [int_to_card(c) for c in community_tuple]

    opp_hole_count = len(my_cards_tuple) 
    need = board_needed + opp_hole_count

    for _ in range(n_sim):
        if len(deck_strs) < need:
            break
            
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
class PlayerAgent(Agent):

    def __init__(self, stream: bool = True, player_id=None):
        super().__init__(stream, player_id=player_id)
        self.action_types = ActionType
        self.start_time = time.perf_counter()
        self.time_limit = 450.0 
        self.total_hands = 1000
        self.net_chips = 0.0
        self.is_guaranteed_win = False
        self.my_total_think_time = 0.0

    def __name__(self):
        return "PlayerAgent"

    def act(self, observation, reward, terminated, truncated, info):
        act_start_time = time.perf_counter()
        # self.logger.info(f"Info Keys: {info.keys()}, Reward: {reward}")
        # self.logger.info(f"Hand {info.get('hand_number', '?')} street {observation['street']}")

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
        
        # Lock in a match lead once the remaining worst-case blind loss cannot erase it.
        if not self.is_guaranteed_win:
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            if self.net_chips > max_possible_loss:
                self.is_guaranteed_win = True
                self.logger.info(
                    "Lead lock enabled: net chips %.2f exceeds max remaining blind loss %.2f.",
                    self.net_chips,
                    max_possible_loss,
                )

        if self.is_guaranteed_win:
            if valid_actions[ActionType.DISCARD.value]:
                return do_discard(0, 1)
            if valid_actions[ActionType.FOLD.value]:
                return fold()
            return check()

        time_remaining = max(0.1, self.time_limit - self.my_total_think_time)

        early_cost_target = 0.85
        late_cost_target = 0.15

        if current_hand <= 400:
            early_hands_left = 400 - current_hand + 1
            late_hands_left = self.total_hands - 400
            base_n_sim = {0: 100, 1: 150, 2: 200, 3: 250}.get(street, 100)
        else:
            early_hands_left = 0
            late_hands_left = self.total_hands - current_hand + 1
            base_n_sim = {0: 50, 1: 100, 2: 150, 3: 200}.get(street, 50)

        total_needed_budget = (early_hands_left * early_cost_target) + (late_hands_left * late_cost_target)
        sim_multiplier = time_remaining / max(1.0, total_needed_budget)

        if sim_multiplier < 0.9:
            self.logger.warning(f"Time budget fall behind, reduce speed multiplier: {sim_multiplier:.2f}")
        
        sim_multiplier = max(0.2, min(2.0, sim_multiplier))

        n_sim = int(base_n_sim * sim_multiplier)
        n_sim = max(10, n_sim)
        
        self.logger.info(f"Hand {current_hand} | Street {street} | Time left: {time_remaining:.1f}s | multiplier: {sim_multiplier:.2f} | n_sim: {n_sim}")

        if valid_actions[ActionType.DISCARD.value]:
            if len(my_cards) == 5:
                best_idx, _ = choose_discard(my_cards, community, opp_discarded, my_discarded, n_sim=n_sim)
                i, j = best_idx
            else:
                i, j = 0, 1
            return do_discard(i, j)
        
        if len(my_cards) < 2:
            return check() if can_check else fold()

        if len(my_cards) == 5:
            _, strength = choose_discard(my_cards, community, opp_discarded, my_discarded, n_sim = n_sim)
        else:
            strength = monte_carlo_strength(my_cards, community, opp_discarded, my_discarded, n_sim)

        opp_aggressive = (opp_bet > my_bet and opp_bet > 4)
        adj = strength - (0.05 if opp_aggressive else 0.0)

        pot_odds = call_amount / (pot + call_amount + 1e-9)

        if call_amount > 35:
            risk_premium = (call_amount / 100.0) * 0.15 
            safe_call_threshold = pot_odds + risk_premium
            safe_call_threshold = min(0.70, safe_call_threshold)
            
            if adj < safe_call_threshold:
                self.logger.info(
                    "Large-bet shield: call=%s, adjusted strength=%.2f, threshold=%.2f",
                    call_amount,
                    adj,
                    safe_call_threshold,
                )
                return check() if can_check else fold()

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
        """Track match-level reward and end-of-hand state."""
        if reward is not None and reward != 0.0:
            self.net_chips += reward
            self.logger.info(f"Getting current game reward : {reward}, Overall reward: {self.net_chips}")

        if terminated:
            current_hand = info.get('hand_number', 1)
            hands_left = self.total_hands - current_hand
            max_possible_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
            self.logger.info(
                "Hand %s finished | total earnings: %.2f | max remaining blind loss: %.2f",
                current_hand,
                self.net_chips,
                max_possible_loss,
            )
