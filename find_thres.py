import random
import time
from gym_env import PokerEnv, WrappedEval
import numpy as np
from functools import lru_cache 
import itertools

int_to_card = PokerEnv.int_to_card
evaluator = WrappedEval()

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


# 載入你寫好的演算法 (假設你的主程式叫做 player.py)
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

def run_threshold_profiling(n_games=2000, mc_sims=150):
    print(f"啟動勝率分佈測量... 總共模擬 {n_games} 局 (MC 次數: {mc_sims})")
    
    # 記錄四個 Street 的勝率分佈
    street_strengths = {0: [], 1: [], 2: [], 3: []}
    
    start_time = time.perf_counter()
    
    for game in range(n_games):
        if game > 0 and game % 100 == 0:
            print(f"進度: {game} / {n_games} 局...")

        # 1. 準備 27 張牌庫並洗牌
        deck = list(range(27))
        random.shuffle(deck)
        
        # 2. 抽牌 (發給 P1 5張, P2 5張, 公共牌 5張)
        p1_hole = deck[0:5]
        p2_hole = deck[5:10]
        board = deck[10:15]
        
        # ==========================================
        # Street 0 (Pre-flop): 0 張公共牌
        # ==========================================
        # 評估 P1 此時 5 張牌的預期最佳勝率
        _, s0_str = choose_discard(p1_hole, [], [], [], n_sim=mc_sims)
        street_strengths[0].append(s0_str)
        
        # ==========================================
        # Street 1 (Flop & Discard): 3 張公共牌 + 換牌
        # ==========================================
        flop_board = board[:3]
        
        # 模擬雙方進行換牌決策
        p1_idx, _ = choose_discard(p1_hole, flop_board, [], [], n_sim=mc_sims)
        p1_keep = [p1_hole[p1_idx[0]], p1_hole[p1_idx[1]]]
        p1_disc = [c for c in p1_hole if c not in p1_keep]
        
        # P2 換牌 (使用相同的邏輯模擬對手)
        p2_idx, _ = choose_discard(p2_hole, flop_board, [], [], n_sim=mc_sims)
        p2_keep = [p2_hole[p2_idx[0]], p2_hole[p2_idx[1]]]
        p2_disc = [c for c in p2_hole if c not in p2_keep]
        
        # 評估 P1 換牌後的勝率 (已知對手棄牌 p2_disc)
        s1_str = monte_carlo_strength(p1_keep, flop_board, p2_disc, p1_disc, n_sim=mc_sims)
        street_strengths[1].append(s1_str)
        
        # ==========================================
        # Street 2 (Turn): 4 張公共牌
        # ==========================================
        turn_board = board[:4]
        s2_str = monte_carlo_strength(p1_keep, turn_board, p2_disc, p1_disc, n_sim=mc_sims)
        street_strengths[2].append(s2_str)
        
        # ==========================================
        # Street 3 (River): 5 張公共牌
        # ==========================================
        river_board = board[:5]
        s3_str = monte_carlo_strength(p1_keep, river_board, p2_disc, p1_disc, n_sim=mc_sims)
        street_strengths[3].append(s3_str)

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ 測量完成！耗時 {elapsed:.1f} 秒。")
    print("="*50)
    print("🏆 27張牌特殊賽制 - 勝率分佈與閾值建議 🏆")
    print("="*50)
    
    # 計算並印出百分位數
    percentiles = [50, 60, 70, 80, 90, 95, 98]
    for street in range(4):
        data = street_strengths[street]
        print(f"\n[ Street {street} ] ({'Pre-flop' if street==0 else 'Flop' if street==1 else 'Turn' if street==2 else 'River'})")
        
        p_values = np.percentile(data, percentiles)
        for p, val in zip(percentiles, p_values):
            # 轉換為容易理解的說法
            tier = ""
            if p == 50: tier = "(中庸牌，超過 50% 的牌)"
            elif p == 80: tier = "(強牌，可以 Call / 小 Raise)"
            elif p == 90: tier = "(超強牌，可以重 Raise)"
            elif p == 98: tier = "(極限堅果牌，All-in 級別)"
            
            print(f"  PR {p:2d} (前 {100-p:2d}% 的牌) -> 勝率閾值: {val:.3f} {tier}")

if __name__ == "__main__":
    # 如果你的電腦跑很慢，可以把 2000 調成 1000
    run_threshold_profiling(n_games=2000, mc_sims=150)