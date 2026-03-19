import random
import itertools
import pickle
import time
import os
import multiprocessing as mp
from gym_env import PokerEnv, WrappedEval
from tqdm import tqdm

# ==========================================
# 全域變數 (每個 CPU 核心各自擁有一份獨立的副本)
# ==========================================
L7 = None  # 7 張牌絕對牌力表
L5 = {}    # 5 張牌當前牌力表
L6 = {}    # 6 張牌當前牌力表

def init_worker():
    """
    這個函數會在每個 CPU 核心啟動時執行一次。
    """
    global L7, L5, L6
    
    # 1. 載入我們之前辛苦算好的 7 張牌字典
    try:
        with open("lookup_table_7cards.pkl", "rb") as f:
            L7 = pickle.load(f)
    except Exception as e:
        print(f"Worker 讀取 7 張牌字典失敗: {e}")
        return

    # 2. 建立 5 張牌與 6 張牌的查表
    evaluator = WrappedEval()
    deck = list(range(27))
    
    # 建立 5 張牌字典
    for combo in itertools.combinations(deck, 5):
        treys_combo = [PokerEnv.int_to_card(c) for c in combo]
        h = treys_combo[:2]
        b = treys_combo[2:]
        L5[tuple(sorted(combo))] = evaluator.evaluate(h, b)
        
    # 建立 6 張牌字典
    for combo in itertools.combinations(deck, 6):
        treys_combo = [PokerEnv.int_to_card(c) for c in combo]
        h = treys_combo[:2]
        b = treys_combo[2:]
        L6[tuple(sorted(combo))] = evaluator.evaluate(h, b)

def process_batch(batch):
    global L7, L5, L6
    results = {}
    
    for H, C in batch:
        HC = tuple(sorted(H + C))
        is_flop = (len(C) == 3)
        known_cards = set(HC)
        unknown_cards = [c for c in range(27) if c not in known_cards]
        
        HP = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        totals = [0, 0, 0]
        
        curr_dict = L5 if is_flop else L6
        hero_curr_score = curr_dict[HC]
        board_needed = 2 if is_flop else 1
        
        # 1. 窮舉對手所有的 5 張牌組合 (Flop有26334種，Turn有20349種)
        for opp_5_cards in itertools.combinations(unknown_cards, 5):
            
            # 2. 對手在這 5 張牌中，找出與公牌搭配「當下最強」的 2 張保留
            best_opp_score = float('inf')
            best_opp_2_cards = None
            
            for opp_candidate in itertools.combinations(opp_5_cards, 2):
                score = curr_dict[tuple(sorted(list(opp_candidate) + list(C)))]
                if score < best_opp_score:
                    best_opp_score = score
                    best_opp_2_cards = opp_candidate
            
            # 3. 比較當前勝率
            if hero_curr_score < best_opp_score:
                curr_idx = 0 
            elif hero_curr_score == best_opp_score:
                curr_idx = 1 
            else:
                curr_idx = 2 
                
            totals[curr_idx] += 1
            
            # 4. 窮舉未來的公牌
            rem_deck = [c for c in unknown_cards if c not in opp_5_cards]
            
            # 這裡不抽樣了，直接暴力窮舉所有未來！(Flop 136種, Turn 16種)
            for F in itertools.combinations(rem_deck, board_needed):
                hero_final_score = L7[tuple(sorted(HC + tuple(F)))]
                opp_final_score = L7[tuple(sorted(list(best_opp_2_cards) + list(C) + list(F)))]
                
                if hero_final_score < opp_final_score:
                    final_idx = 0
                elif hero_final_score == opp_final_score:
                    final_idx = 1
                else:
                    final_idx = 2
                HP[curr_idx][final_idx] += 1
                
        # --- 計算指標 ---
        total_opp_hands = sum(totals) # Flop 必定是 26334，Turn 必定是 20349
        if total_opp_hands == 0:
            results[(H, C)] = (0.5, 0.0, 0.0)
            continue
            
        # 計算當下勝率
        HS = (totals[0] + 0.5 * totals[1]) / total_opp_hands
        
        # 計算未來公牌的組合數，用來把膨脹的分子除回來 (解決維度不對齊 Bug)
        # Flop: C(17,2) = 136 | Turn: C(16,1) = 16
        future_combos = 136 if is_flop else 16
        
        denom_ppot = totals[2] + 0.5 * totals[1]
        PPot = (HP[2][0] + 0.5 * HP[2][1] + 0.5 * HP[1][0]) / (denom_ppot * future_combos) if denom_ppot > 0 else 0.0
            
        denom_npot = totals[0] + 0.5 * totals[1]
        NPot = (HP[0][2] + 0.5 * HP[0][1] + 0.5 * HP[1][2]) / (denom_npot * future_combos) if denom_npot > 0 else 0.0
            
        results[(H, C)] = (HS, PPot, NPot)
        
    return results

def generate_states_and_compute(is_flop=True):
    stage_name = "Flop" if is_flop else "Turn"
    num_comm_cards = 3 if is_flop else 4
    deck = list(range(27))
    
    print(f"\n[準備 {stage_name} 任務...]")
    all_states_set = set()
    
    for combo in itertools.combinations(deck, 2 + num_comm_cards):
        for h_indices in itertools.combinations(range(len(combo)), 2):
            H = tuple(sorted([combo[i] for i in h_indices]))
            C = tuple(sorted([combo[i] for i in range(len(combo)) if i not in h_indices]))
            all_states_set.add((H, C))
            
    all_states = list(all_states_set)
    total_states = len(all_states)
    
    # 將任務切塊
    chunk_size = 4
    chunks = [all_states[i:i + chunk_size] for i in range(0, total_states, chunk_size)]
    
    final_dict = {}
    start_time = time.time()
    
    # 使用 tqdm 顯示進度
    # desc: 進度條前的描述文字
    # total: 總共要跑多少次 (這裡我們跑的是 chunks 的數量)
    # unit: 單位名稱
    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:
        pbar = tqdm(pool.imap_unordered(process_batch, chunks), 
                    total=len(chunks), 
                    desc=f"🚀 計算 {stage_name} EHS", 
                    unit="chunk",
                    leave=True)
        
        for result_dict in pbar:
            final_dict.update(result_dict)
                
    total_time = time.time() - start_time
    print(f"✅ {stage_name} 完成！總耗時 {total_time:.1f} 秒，共 {len(final_dict)} 筆。")
    
    return final_dict

if __name__ == "__main__":
    # 1. Flop
    flop_ehs_dict = generate_states_and_compute(is_flop=True)
    with open("flop_ehs_table_opponent.pkl", "wb") as f:
        pickle.dump(flop_ehs_dict, f)
    
    # 2. Turn
    turn_ehs_dict = generate_states_and_compute(is_flop=False)
    with open("turn_ehs_table_opponent.pkl", "wb") as f:
        pickle.dump(turn_ehs_dict, f)
    
    # 3. 合併
    combined_ehs = {**flop_ehs_dict, **turn_ehs_dict}
    with open("EHS_table_final_opponent.pkl", "wb") as f:
        pickle.dump(combined_ehs, f)
    print(f"🎉 任務完成！最終存儲筆數: {len(combined_ehs)}")