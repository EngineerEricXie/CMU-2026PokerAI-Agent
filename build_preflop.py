import itertools
import random
import pickle
import time
import os
from tqdm import tqdm

def build_preflop_table():
    print("載入 7 張牌基礎牌力表以加速模擬...")
    with open("lookup_table_7cards.pkl", "rb") as f:
        lookup_table_7 = pickle.load(f)

    deck_ints = list(range(27))
    all_starting_hands = list(itertools.combinations(deck_ints, 5))
    total_hands = len(all_starting_hands)
    
    preflop_equity_table = {}
    N_SIMULATIONS = 1000 # 模擬 1000 次，取得足夠準確的勝率
    
    print(f"開始建構 Pre-flop 期望值表！共 {total_hands} 種起手牌。")
    start_time = time.time()

    # for idx, hero_cards in enumerate(all_starting_hands):
    for idx, hero_cards in tqdm(enumerate(all_starting_hands), total=total_hands, desc="建構進度"):
        if idx % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"進度: {idx} / {total_hands} (耗時 {elapsed:.1f} 秒)...")
            
        hero_cards_set = set(hero_cards)
        rem_deck = [c for c in deck_ints if c not in hero_cards_set]
        wins = ties = 0
        
        for _ in range(N_SIMULATIONS):
            sampled = random.sample(rem_deck, 10)
            villain_cards = sampled[:5]
            board_cards = sampled[5:]
            
            hero_best_score = float('inf')
            for keep_2 in itertools.combinations(hero_cards, 2):
                # 利用 7 張牌字典秒算最佳組合
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
                
        preflop_equity_table[hero_cards] = (wins + 0.5 * ties) / N_SIMULATIONS

    with open("preflop_table.pkl", "wb") as f:
        pickle.dump(preflop_equity_table, f)
    print("存檔成功：preflop_table.pkl")

if __name__ == "__main__":
    build_preflop_table()