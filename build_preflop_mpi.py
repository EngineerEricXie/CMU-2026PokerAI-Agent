import itertools
import random
import pickle
import time
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

N_SIMULATIONS = 10000

# ⚠️ global（給 worker 用）
lookup_table_7 = None
deck_ints = list(range(27))


def init_worker():
    global lookup_table_7
    with open("lookup_table_7cards.pkl", "rb") as f:
        lookup_table_7 = pickle.load(f)


def simulate_one_hand(hero_cards):
    hero_cards_set = set(hero_cards)
    rem_deck = [c for c in deck_ints if c not in hero_cards_set]

    wins = ties = 0

    for _ in range(N_SIMULATIONS):
        sampled = random.sample(rem_deck, 10)
        villain_cards = sampled[:5]
        board_cards = sampled[5:]

        hero_best_score = float('inf')
        for keep_2 in itertools.combinations(hero_cards, 2):
            score = lookup_table_7[tuple(sorted(list(keep_2) + board_cards))]
            if score < hero_best_score:
                hero_best_score = score

        villain_best_score = float('inf')
        for keep_2 in itertools.combinations(villain_cards, 2):
            score = lookup_table_7[tuple(sorted(list(keep_2) + board_cards))]
            if score < villain_best_score:
                villain_best_score = score

        if hero_best_score < villain_best_score:
            wins += 1
        elif hero_best_score == villain_best_score:
            ties += 1

    equity = (wins + 0.5 * ties) / N_SIMULATIONS
    return hero_cards, equity


def build_preflop_table():
    all_starting_hands = list(itertools.combinations(deck_ints, 5))
    total_hands = len(all_starting_hands)

    print(f"開始建構 Pre-flop 表，共 {total_hands} 手牌")
    start_time = time.time()

    num_workers = 8
    print(f"使用核心數: {num_workers}")

    preflop_equity_table = {}

    with Pool(processes=num_workers, initializer=init_worker) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(simulate_one_hand, all_starting_hands), total=total_hands):
            hero_cards, equity = result
            preflop_equity_table[hero_cards] = equity

    with open("preflop_table.pkl", "wb") as f:
        pickle.dump(preflop_equity_table, f)

    print(f"完成！耗時 {time.time() - start_time:.1f} 秒")


if __name__ == "__main__":
    build_preflop_table()