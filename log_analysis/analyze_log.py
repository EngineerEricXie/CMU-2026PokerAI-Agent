import pandas as pd
import ast
import pickle
import os
import numpy as np
import re

# ==========================================
# 1. 遊戲引擎常數與卡牌轉換
# ==========================================
RANKS = "23456789A"  
SUITS = "dhs"        

def card_str_to_int(card_str: str) -> int:
    if not card_str or len(card_str) != 2:
        return -1
    try:
        rank_idx = RANKS.index(card_str[0])
        suit_idx = SUITS.index(card_str[1])
        return suit_idx * len(RANKS) + rank_idx
    except ValueError:
        return -1

# ==========================================
# 2. 載入神級字典
# ==========================================
PREFLOP_TABLE = {}
EHS_TABLE = {}

def load_tables():
    global PREFLOP_TABLE, EHS_TABLE
    if os.path.exists("preflop_table.pkl"):
        with open("preflop_table.pkl", "rb") as f:
            PREFLOP_TABLE = pickle.load(f)
    if os.path.exists("EHS_table_fixed.pkl"):
        with open("EHS_table_fixed.pkl", "rb") as f:
            EHS_TABLE = pickle.load(f)

def get_cards_tuple(cards_str_list: str) -> tuple:
    try:
        cards_list = ast.literal_eval(cards_str_list)
        int_cards = [card_str_to_int(c.strip()) for c in cards_list]
        return tuple(sorted(int_cards))
    except (ValueError, SyntaxError):
        return ()

# ==========================================
# 3. 核心分析引擎
# ==========================================
def analyze_match(csv_file: str):
    print(f"\n🚀 開始解析賽事紀錄: {csv_file}")
    
    # --- 讀取隊伍名稱 ---
    team_names = {0: "Team 0", 1: "Team 1"}
    with open(csv_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line.startswith("#"):
            match = re.search(r"Team 0:\s*([^,]+),\s*Team 1:\s*(.+)", first_line)
            if match:
                team_names[0] = match.group(1).strip()
                team_names[1] = match.group(2).strip()

    # --- 讀取 DataFrame ---
    df = pd.read_csv(csv_file, comment='#')
    df.columns = df.columns.str.strip()
    
    stats = {
        0: {'hands_played': 0, 'vpip': 0, 'pfr': 0, 'calls': 0, 'raises': 0, 
            'won_by_fold': 0, 'profit_by_fold': 0, 'fold_to_big_raise': 0, 'fold_to_small_bet': 0,
            'won_at_showdown': 0, 'profit_at_showdown': 0, 'showdown_big_win': 0, 'showdown_small_win': 0,
            'bluff_raises': 0, 'value_raises': 0, 'ehs_when_raising': []},
        1: {'hands_played': 0, 'vpip': 0, 'pfr': 0, 'calls': 0, 'raises': 0, 
            'won_by_fold': 0, 'profit_by_fold': 0, 'fold_to_big_raise': 0, 'fold_to_small_bet': 0,
            'won_at_showdown': 0, 'profit_at_showdown': 0, 'showdown_big_win': 0, 'showdown_small_win': 0,
            'bluff_raises': 0, 'value_raises': 0, 'ehs_when_raising': []}
    }
    
    hand_groups = df.groupby('hand_number')
    hand_numbers = list(hand_groups.groups.keys())
    
    for i, hand_num in enumerate(hand_numbers):
        hand_data = hand_groups.get_group(hand_num)
        stats[0]['hands_played'] += 1
        stats[1]['hands_played'] += 1
        
        preflop_data = hand_data[hand_data['street'] == 'Pre-Flop']
        for team in [0, 1]:
            team_preflop = preflop_data[preflop_data['active_team'] == team]
            if any(team_preflop['action_type'].isin(['CALL', 'RAISE'])):
                stats[team]['vpip'] += 1
            if any(team_preflop['action_type'] == 'RAISE'):
                stats[team]['pfr'] += 1
                
        for _, row in hand_data.iterrows():
            team = row['active_team']
            action = row['action_type']
            street = row['street']
            
            if action == 'CALL':
                stats[team]['calls'] += 1
            elif action == 'RAISE':
                stats[team]['raises'] += 1
                
                team_cards = row[f'team_{team}_cards']
                board = row['board_cards']
                
                if street == 'Pre-Flop' and PREFLOP_TABLE:
                    cards_tuple = get_cards_tuple(team_cards)
                    if cards_tuple in PREFLOP_TABLE:
                        equity = PREFLOP_TABLE[cards_tuple]
                        stats[team]['ehs_when_raising'].append(PREFLOP_TABLE[cards_tuple])
                        # --- 新增這段分類邏輯 ---
                        # 翻前勝率低於 0.45 卻主動加注，視為翻前詐唬 (Loose Raise)
                        if equity < 0.45:
                            stats[team]['bluff_raises'] += 1
                        else:
                            stats[team]['value_raises'] += 1
                        
                elif street in ['Flop', 'Turn'] and EHS_TABLE:
                    h_tuple = get_cards_tuple(team_cards)
                    c_tuple = get_cards_tuple(board)
                    if h_tuple and c_tuple:
                        # 【關鍵修復】: EHS 字典的 Key 是 (底牌Tuple, 公牌Tuple)
                        combo_key = (h_tuple, c_tuple)
                        if combo_key in EHS_TABLE:
                            HS, PPot, NPot = EHS_TABLE[combo_key]
                            ehs_val = HS + (1 - HS) * PPot - HS * NPot
                            stats[team]['ehs_when_raising'].append(ehs_val)
                            
                            if ehs_val < 0.45:
                                stats[team]['bluff_raises'] += 1
                            else:
                                stats[team]['value_raises'] += 1

        # --- 分析結局與獲利特徵 ---
        last_action = hand_data.iloc[-1]
        if last_action['action_type'] == 'FOLD':
            loser = last_action['active_team']
            winner = 1 - loser
            chips_won = last_action[f'team_{loser}_bet'] 
            
            stats[winner]['won_by_fold'] += 1
            stats[winner]['profit_by_fold'] += chips_won
            
            # 判斷對手是屈服於大加注還是小下注 (大於 15 籌碼視為大壓力)
            pressure = last_action[f'team_{winner}_bet'] - last_action[f'team_{loser}_bet']
            if pressure >= 15:
                stats[winner]['fold_to_big_raise'] += 1
            else:
                stats[winner]['fold_to_small_bet'] += 1
        else:
            if i + 1 < len(hand_numbers):
                next_hand = hand_groups.get_group(hand_numbers[i+1]).iloc[0]
                delta_0 = next_hand['team_0_bankroll'] - hand_data.iloc[0]['team_0_bankroll']
                
                if delta_0 > 0:
                    stats[0]['won_at_showdown'] += 1
                    stats[0]['profit_at_showdown'] += delta_0
                    if delta_0 >= 35: stats[0]['showdown_big_win'] += 1
                    else: stats[0]['showdown_small_win'] += 1
                elif delta_0 < 0:
                    stats[1]['won_at_showdown'] += 1
                    stats[1]['profit_at_showdown'] += abs(delta_0)
                    if abs(delta_0) >= 35: stats[1]['showdown_big_win'] += 1
                    else: stats[1]['showdown_small_win'] += 1

    # ==========================================
    # 4. 產出報告
    # ==========================================
    print("\n" + "="*70)
    print("🏆 賽後對手風格與獲利特徵分析 🏆")
    print("="*70)
    
    for team in [0, 1]:
        s = stats[team]
        hands = max(1, s['hands_played'])
        name = team_names[team]
        
        vpip_pct = (s['vpip'] / hands) * 100
        pfr_pct = (s['pfr'] / hands) * 100
        af = s['raises'] / max(1, s['calls']) if s['calls'] > 0 else float('inf')
        
        avg_raise_ehs = np.mean(s['ehs_when_raising']) if s['ehs_when_raising'] else 0.0
        
        print(f"\n🔹 【Team {team}: {name}】 ({hands} Hands Played)")
        print("-" * 50)
        print(f"📊 基礎風格 (VPIP: {vpip_pct:.1f}% | PFR: {pfr_pct:.1f}% | AF 激進度: {af:.2f})")
        
        print(f"\n💰 獲利結構剖析 (總計: {s['profit_by_fold'] + s['profit_at_showdown']} 籌碼):")
        print(f"  [不戰而勝 (Won by Fold)] 共贏 {s['profit_by_fold']} 籌碼 / {s['won_by_fold']} 次")
        print(f"    ↳ 靠【巨大加注】逼退對手: {s['fold_to_big_raise']} 次")
        print(f"    ↳ 靠【微小下注】勸退對手: {s['fold_to_small_bet']} 次")
        
        print(f"  [攤牌比大小 (Showdown)] 共贏 {s['profit_at_showdown']} 籌碼 / {s['won_at_showdown']} 次")
        print(f"    ↳ 贏下大底池 (>35): {s['showdown_big_win']} 次")
        print(f"    ↳ 贏下小底池 (<=35): {s['showdown_small_win']} 次")
        
        print(f"\n🕵️ 核心加注品質 (Avg EHS: {avg_raise_ehs:.2f}):")
        total_classified = s['bluff_raises'] + s['value_raises']
        if total_classified > 0:
            bluff_pct = (s['bluff_raises'] / total_classified) * 100
            print(f"  • 純價值打法: {100 - bluff_pct:.1f}% | 詐唬/半詐唬: {bluff_pct:.1f}%")
        else:
            print("  • 該玩家無足夠加注樣本進行 EHS 分析。")

if __name__ == "__main__":
    load_tables()
    # 執行分析
    analyze_match('match.csv')