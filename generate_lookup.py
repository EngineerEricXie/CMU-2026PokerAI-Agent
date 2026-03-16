import json
import itertools
import random
from gym_env import PokerEnv

def simulate_preflop_equity(card1, card2, num_simulations=5000):
    """對特定的兩張起手牌進行高精度的蒙地卡羅模擬"""
    deck = set(range(27))
    my_cards = {card1, card2}
    non_shown_cards = list(deck - my_cards)
    
    wins = 0
    evaluator = PokerEnv().evaluator
    my_hand = [PokerEnv.int_to_card(card1), PokerEnv.int_to_card(card2)]
    
    for _ in range(num_simulations):
        # 隨機抽出對手的 2 張牌與 5 張公牌
        sample = random.sample(non_shown_cards, 7)
        opp_hand = [PokerEnv.int_to_card(c) for c in sample[:2]]
        board = [PokerEnv.int_to_card(c) for c in sample[2:]]
        
        my_rank = evaluator.evaluate(my_hand, board)
        opp_rank = evaluator.evaluate(opp_hand, board)
        if my_rank < opp_rank: # 數字越小牌型越大
            wins += 1
            
    return wins / num_simulations

if __name__ == "__main__":
    print("開始生成 27 張牌變體的翻牌前勝率表...")
    equity_table = {}
    # 遍歷所有 351 種起手牌組合
    for c1, c2 in itertools.combinations(range(27), 2):
        key = f"{min(c1, c2)}_{max(c1, c2)}"
        equity = simulate_preflop_equity(c1, c2, num_simulations=5000)
        equity_table[key] = equity
        print(f"Cards ({c1}, {c2}) Equity: {equity:.4f}")

    with open("preflop_equity.json", "w") as f:
        json.dump(equity_table, f, indent=4)
    print("✅ preflop_equity.json 生成完畢！")