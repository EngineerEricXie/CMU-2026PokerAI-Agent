import random
import itertools
from collections import Counter

from agents.agent import Agent
from gym_env import PokerEnv

ActionType = PokerEnv.ActionType

# ---------------------------------------------------------------------------
# Card helpers
# Card encoding: card = rank_index * 3 + suit_index
# Ranks: "23456789A" (index 0-8)
# Suits: "dhs"       (index 0-2)
# ---------------------------------------------------------------------------

def rank(card: int) -> int:
    return card // 3

def suit(card: int) -> int:
    return card % 3

def valid_cards(card_tuple) -> list:
    return [c for c in card_tuple if c >= 0]

# ---------------------------------------------------------------------------
# Hand evaluator (pure Python, no external libs)
# Returns a score tuple where LOWER = BETTER hand
# Categories: 1=StraightFlush, 2=FullHouse, 3=Flush, 4=Straight,
#             5=ThreeOfAKind, 6=TwoPair, 7=OnePair, 8=HighCard
# ---------------------------------------------------------------------------

def _score_five(cards: list) -> tuple:
    """Score exactly 5 cards. Lower tuple = better hand."""
    ranks = sorted([rank(c) for c in cards], reverse=True)
    suits = [suit(c) for c in cards]

    flush = (len(set(suits)) == 1)

    # Straight detection (normal + A-low: A=8, 2-5 = indices 0-3)
    def is_straight(r):
        if len(set(r)) == 5 and r[0] - r[4] == 4:
            return True, r[0]
        if sorted(r) == [0, 1, 2, 3, 8]:   # A-2-3-4-5
            return True, 3                   # high card = 5 (index 3)
        return False, None

    straight, s_high = is_straight(ranks)

    cnt = Counter(ranks)
    # Sort by (frequency desc, rank desc) for tiebreaking
    freq = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    group_sizes = [f[1] for f in freq]
    group_ranks = [f[0] for f in freq]

    if flush and straight:
        return (1, s_high)
    if group_sizes[0] == 3 and len(group_sizes) > 1 and group_sizes[1] == 2:
        return (2, group_ranks[0], group_ranks[1])
    if flush:
        return (3, tuple(ranks))
    if straight:
        return (4, s_high)
    if group_sizes[0] == 3:
        return (5, group_ranks[0], tuple(group_ranks[1:]))
    if group_sizes[0] == 2 and len(group_sizes) > 1 and group_sizes[1] == 2:
        top2 = sorted([group_ranks[0], group_ranks[1]], reverse=True)
        kicker = group_ranks[2] if len(group_ranks) > 2 else 0
        return (6, top2[0], top2[1], kicker)
    if group_sizes[0] == 2:
        return (7, group_ranks[0], tuple(group_ranks[1:]))
    return (8, tuple(ranks))


def evaluate_best_hand(hole: list, board: list) -> tuple:
    """
    Best 5-card hand from hole cards + board.
    Works with 2-7 total cards.
    """
    all_cards = hole + board
    if len(all_cards) < 5:
        # Not enough cards yet - use partial rank estimate
        ranks = sorted([rank(c) for c in all_cards], reverse=True)
        cnt = Counter(ranks)
        groups = sorted(cnt.values(), reverse=True)
        if groups[0] >= 2:
            return (7, ranks[0], tuple(ranks[1:]))
        return (8, tuple(ranks))
    # Try all 5-card combos, take best (lowest)
    best = None
    for combo in itertools.combinations(all_cards, 5):
        s = _score_five(list(combo))
        if best is None or s < best:
            best = s
    return best


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def monte_carlo_strength(
    my_cards: list,
    community: list,
    opp_discarded: list,
    n_sim: int = 200
) -> float:
    """
    Estimate P(win) by simulating random completions of the board
    and random opponent hole cards.
    Accounts for known opponent discards (they can't be in opp's hand).
    """
    known = set(my_cards) | set(community) | set(opp_discarded)
    deck = [c for c in range(27) if c not in known]

    board_needed = 5 - len(community)
    wins = ties = total = 0

    for _ in range(n_sim):
        need = board_needed + 2
        if len(deck) < need:
            break
        sample = random.sample(deck, need)
        sim_board = community + sample[:board_needed]
        opp_hole  = sample[board_needed:]

        my_score  = evaluate_best_hand(my_cards, sim_board)
        opp_score = evaluate_best_hand(opp_hole,  sim_board)

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
    n_sim: int = 100
) -> tuple:
    """
    Try all C(5,2)=10 ways to keep 2 cards from 5 hole cards.
    Return (i, j) indices of the best pair to keep.
    """
    best_idx = (0, 1)
    best_str = -1.0
    for i, j in itertools.combinations(range(5), 2):
        kept = [hole[i], hole[j]]
        s = monte_carlo_strength(kept, community, opp_discarded, n_sim)
        if s > best_str:
            best_str = s
            best_idx = (i, j)
    return best_idx


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
                i, j = choose_discard(my_cards, community, opp_discarded, n_sim=120)
            else:
                i, j = 0, 1
            return ActionType.DISCARD.value, 0, i, j

        # ── Safety check ──────────────────────────────────────────────
        if len(my_cards) < 2:
            return check() if can_check else fold()

        # ── Estimate hand strength ─────────────────────────────────────
        # More simulations on later streets (more info, worth the compute)
        n_sim = {0: 150, 1: 200, 2: 250, 3: 300}.get(street, 150)
        strength = monte_carlo_strength(my_cards, community, opp_discarded, n_sim)

        # Slight penalty when facing aggression (opponent may have strong hand)
        opp_aggressive = (opp_bet > my_bet and opp_bet > 4)
        adj = strength - (0.05 if opp_aggressive else 0.0)

        # Pot odds needed to profitably call
        pot_odds = call_amount / (pot + call_amount + 1e-9)

        self.logger.info(
            f"  strength={strength:.3f} adj={adj:.3f} pot_odds={pot_odds:.3f} "
            f"call={call_amount} pot={pot}"
        )

        # ── Betting decisions ──────────────────────────────────────────

        # Very strong hand (>75%): raise, sized by how strong
        if adj > 0.75:
            if can_raise:
                frac = min(1.0, 0.5 + (adj - 0.75) * 2.0)
                return make_raise(frac)
            if can_call:  return call()
            return check()

        # Good hand (60-75%): raise small or call if odds are right
        if adj > 0.60:
            if can_raise and adj > 0.68:
                return make_raise(0.25)
            if can_call and adj > pot_odds + 0.05:
                return call()
            if can_check: return check()
            if can_call:  return call()
            return fold()

        # Marginal hand (42-60%): check for free or cheap call only
        if adj > 0.42:
            if can_check: return check()
            if can_call and call_amount <= max(2, int(pot * 0.25)):
                return call()
            return fold()

        # Weak hand (<42%): take the free check or fold
        if can_check: return check()
        return fold()