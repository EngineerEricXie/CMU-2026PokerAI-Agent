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

def get_ehs(team_cards_str, board_str, street):
    """給定手牌與公牌字串，回傳 EHS 值。失敗回傳 None。"""
    if street == 'Pre-Flop' and PREFLOP_TABLE:
        cards_tuple = get_cards_tuple(team_cards_str)
        if cards_tuple in PREFLOP_TABLE:
            return PREFLOP_TABLE[cards_tuple]
    elif street in ['Flop', 'Turn'] and EHS_TABLE:
        h_tuple = get_cards_tuple(team_cards_str)
        c_tuple = get_cards_tuple(board_str)
        if h_tuple and c_tuple:
            combo_key = (h_tuple, c_tuple)
            if combo_key in EHS_TABLE:
                HS, PPot, NPot = EHS_TABLE[combo_key]
                return HS + (1 - HS) * PPot - HS * NPot
    return None

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
    
    hand_groups = df.groupby('hand_number')
    hand_numbers = list(hand_groups.groups.keys())
    total_hands = max(hand_numbers)
    
    # --- 偵測必勝鎖定 (Guaranteed Win) ---
    lock_hand = total_hands + 1
    locking_team = None
    
    for i, hand_num in enumerate(hand_numbers):
        hand_data = hand_groups.get_group(hand_num)
        start_row = hand_data.iloc[0]
        
        hands_left = total_hands - hand_num + 1
        max_loss = (hands_left // 2) * 3 + (2 if hands_left % 2 != 0 else 0)
        
        if start_row.get('team_0_bankroll', 0) > max_loss:
            lock_hand = hand_num
            locking_team = 0
            break
        elif start_row.get('team_1_bankroll', 0) > max_loss:
            lock_hand = hand_num
            locking_team = 1
            break

    if locking_team is not None:
        print(f"🚨 【系統偵測】Team {locking_team} ({team_names[locking_team]}) 在第 {lock_hand} 局觸發「必勝鎖定」！後續進入垃圾時間。")
    else:
        print("⚔️ 【系統偵測】雙方血戰到底，無人觸發必勝鎖定。")

    # --- 數據結構 ---
    stats = {
        0: {'hands_played': 0, 'vpip': 0, 'pfr': 0, 'calls': 0, 'raises': 0, 
            'saw_flop': 0, 'went_to_showdown': 0, 'won_at_showdown': 0,
            'won_by_fold': 0, 'profit_by_fold': 0, 'profit_at_showdown': 0,
            'bluff_raises': 0, 'value_raises': 0, 'ehs_when_raising': [],
            'flop_raises': 0, 'flop_calls': 0,
            'discard_kept_ehs': [],
            # 新增：攤牌輸的局，最後一次加注的 EHS
            'ehs_on_lost_showdown': [],
            'ehs_on_won_showdown': []},
        1: {'hands_played': 0, 'vpip': 0, 'pfr': 0, 'calls': 0, 'raises': 0, 
            'saw_flop': 0, 'went_to_showdown': 0, 'won_at_showdown': 0,
            'won_by_fold': 0, 'profit_by_fold': 0, 'profit_at_showdown': 0,
            'bluff_raises': 0, 'value_raises': 0, 'ehs_when_raising': [],
            'flop_raises': 0, 'flop_calls': 0,
            'discard_kept_ehs': [],
            'ehs_on_lost_showdown': [],
            'ehs_on_won_showdown': []}
    }
    
    # --- 統計開始 ---
    for i, hand_num in enumerate(hand_numbers):
        if hand_num >= lock_hand:
            break
            
        hand_data = hand_groups.get_group(hand_num)
        hand_rows = list(hand_data.iterrows())

        stats[0]['hands_played'] += 1
        stats[1]['hands_played'] += 1
        
        streets_seen = hand_data['street'].unique()
        if 'Flop' in streets_seen:
            stats[0]['saw_flop'] += 1
            stats[1]['saw_flop'] += 1
            
        last_action = hand_data.iloc[-1]
        if last_action['action_type'] != 'FOLD':
            if 'Flop' in streets_seen:
                stats[0]['went_to_showdown'] += 1
                stats[1]['went_to_showdown'] += 1
        
        # 基本動作分析
        preflop_data = hand_data[hand_data['street'] == 'Pre-Flop']
        for team in [0, 1]:
            team_preflop = preflop_data[preflop_data['active_team'] == team]
            if any(team_preflop['action_type'].isin(['CALL', 'RAISE'])):
                stats[team]['vpip'] += 1
            if any(team_preflop['action_type'] == 'RAISE'):
                stats[team]['pfr'] += 1

        # 追蹤每個 team 本局最後一次加注的 EHS（用於攤牌輸贏分析）
        last_raise_ehs = {0: None, 1: None}

        for idx, (_, row) in enumerate(hand_rows):
            team = row['active_team']
            action = row['action_type']
            street = row['street']
            
            if action == 'CALL':
                stats[team]['calls'] += 1
                if street == 'Flop':
                    stats[team]['flop_calls'] += 1

            elif action == 'RAISE':
                stats[team]['raises'] += 1
                if street == 'Flop':
                    stats[team]['flop_raises'] += 1
                
                team_cards = row[f'team_{team}_cards']
                board = row['board_cards']
                
                ehs_val = get_ehs(team_cards, board, street)
                if ehs_val is not None:
                    stats[team]['ehs_when_raising'].append(ehs_val)
                    last_raise_ehs[team] = ehs_val  # 更新本局最後加注 EHS
                    if ehs_val < 0.45:
                        stats[team]['bluff_raises'] += 1
                    else:
                        stats[team]['value_raises'] += 1

            elif action == 'DISCARD':
                # 取下一個同 team 的動作行，那時才是保留的 2 張牌
                for next_idx in range(idx + 1, len(hand_rows)):
                    next_row = hand_rows[next_idx][1]
                    if next_row['active_team'] == team:
                        kept_cards = next_row[f'team_{team}_cards']
                        board = next_row['board_cards']
                        h_tuple = get_cards_tuple(kept_cards)
                        c_tuple = get_cards_tuple(board)
                        if h_tuple and c_tuple and len(h_tuple) == 2:
                            combo_key = (h_tuple, c_tuple)
                            if combo_key in EHS_TABLE:
                                HS, PPot, NPot = EHS_TABLE[combo_key]
                                ehs_val = HS + (1 - HS) * PPot - HS * NPot
                                stats[team]['discard_kept_ehs'].append(ehs_val)
                        break

        # 勝負與獲利分析
        if last_action['action_type'] == 'FOLD':
            loser = last_action['active_team']
            winner = 1 - loser
            chips_won = last_action[f'team_{loser}_bet']
            stats[winner]['won_by_fold'] += 1
            stats[winner]['profit_by_fold'] += chips_won
        else:
            # 攤牌：判斷誰贏，並記錄雙方最後加注的 EHS
            if i + 1 < len(hand_numbers):
                next_hand = hand_groups.get_group(hand_numbers[i+1]).iloc[0]
                delta_0 = next_hand['team_0_bankroll'] - hand_data.iloc[0]['team_0_bankroll']
                
                if delta_0 > 0:
                    winner, loser = 0, 1
                elif delta_0 < 0:
                    winner, loser = 1, 0
                else:
                    winner, loser = None, None

                if winner is not None:
                    profit = abs(delta_0)
                    stats[winner]['won_at_showdown'] += 1
                    stats[winner]['profit_at_showdown'] += profit

                    # 記錄贏家和輸家在攤牌局的最後加注 EHS
                    if last_raise_ehs[winner] is not None:
                        stats[winner]['ehs_on_won_showdown'].append(last_raise_ehs[winner])
                    if last_raise_ehs[loser] is not None:
                        stats[loser]['ehs_on_lost_showdown'].append(last_raise_ehs[loser])

    # ==========================================
    # 4. 產出報告
    # ==========================================
    print("\n" + "="*75)
    print(f"🏆 真實戰鬥期分析 (第 1 局 ~ 第 {lock_hand - 1} 局) 🏆")
    print("="*75)
    
    for team in [0, 1]:
        s = stats[team]
        hands = max(1, s['hands_played'])
        name = team_names[team]
        
        vpip_pct = (s['vpip'] / hands) * 100
        pfr_pct = (s['pfr'] / hands) * 100
        af = s['raises'] / s['calls'] if s['calls'] > 0 else float(s['raises'])
        flop_af = s['flop_raises'] / s['flop_calls'] if s['flop_calls'] > 0 else float(s['flop_raises'])
        
        wtsd_pct = (s['went_to_showdown'] / s['saw_flop'] * 100) if s['saw_flop'] > 0 else 0
        wsd_pct = (s['won_at_showdown'] / s['went_to_showdown'] * 100) if s['went_to_showdown'] > 0 else 0
        avg_raise_ehs = np.mean(s['ehs_when_raising']) if s['ehs_when_raising'] else 0.0
        avg_discard_ehs = np.mean(s['discard_kept_ehs']) if s['discard_kept_ehs'] else 0.0

        # 攤牌輸贏的 EHS 分析
        avg_ehs_won = np.mean(s['ehs_on_won_showdown']) if s['ehs_on_won_showdown'] else 0.0
        avg_ehs_lost = np.mean(s['ehs_on_lost_showdown']) if s['ehs_on_lost_showdown'] else 0.0
        n_won = len(s['ehs_on_won_showdown'])
        n_lost = len(s['ehs_on_lost_showdown'])

        # 判斷輸牌原因
        if avg_ehs_lost < 0.50:
            loss_diagnosis = "⚠️ 用弱牌硬打被抓（真詐唬失敗）"
        elif avg_ehs_lost < 0.65:
            loss_diagnosis = "🎲 中等牌力進大底池，被更強的牌擊敗（需提高跟注門檻）"
        elif avg_ehs_lost < 0.72:
            loss_diagnosis = "⚡ 中上牌力輸攤牌（策略邊界問題，跟注門檻可再收緊）"
        else:
            loss_diagnosis = "😓 強牌輸攤牌（純運氣，策略無誤）"

        print(f"\n🔹 【Team {team}: {name}】 ({hands} Hands Played)")
        print("-" * 55)
        print(f"📊 基礎風格: VPIP: {vpip_pct:.1f}% | PFR: {pfr_pct:.1f}% | 總 AF: {af:.2f}")
        
        print(f"\n🧐 職業級攤牌與翻後指標:")
        print(f"  • Flop AF: {flop_af:.2f} （此變體翻牌圈強制換牌，Flop AF 參考價值低）")
        print(f"  • WTSD% (看過翻牌後打到攤牌的機率): {wtsd_pct:.1f}%")
        print(f"  • W$SD% (攤牌贏錢率): {wsd_pct:.1f}% (正常值約 50%，>60% 代表只拿天牌去攤牌)")
        print(f"  • 棄牌後平均保留手牌強度 (EHS): {avg_discard_ehs:.2f}  ← 越高代表棄牌決策越好")
        
        print(f"\n💰 真實獲利結構 (總利潤: {s['profit_by_fold'] + s['profit_at_showdown']} 籌碼):")
        print(f"  • 靠逼退對手贏得: {s['profit_by_fold']} 籌碼 ({s['won_by_fold']} 次)")
        print(f"  • 攤牌比大小贏得: {s['profit_at_showdown']} 籌碼 ({s['won_at_showdown']} 次)")
        
        print(f"\n🕵️ 加注品質透視 (Avg EHS: {avg_raise_ehs:.2f}):")
        total_classified = s['bluff_raises'] + s['value_raises']
        if total_classified > 0:
            bluff_pct = (s['bluff_raises'] / total_classified) * 100
            print(f"  • 純價值打法: {100 - bluff_pct:.1f}% | 詐唬/半詐唬: {bluff_pct:.1f}%")
        else:
            print("  • 樣本數不足，無法分析。")

        print(f"\n🔬 攤牌輸贏根因分析:")
        print(f"  • 贏得攤牌時，最後加注平均 EHS: {avg_ehs_won:.2f}  (樣本 {n_won} 局)")
        print(f"  • 輸掉攤牌時，最後加注平均 EHS: {avg_ehs_lost:.2f}  (樣本 {n_lost} 局)")
        print(f"  • 輸牌診斷: {loss_diagnosis}")

if __name__ == "__main__":
    load_tables()
    # for i in range(3):
    #     analyze_match(f'../match{i}.csv')
    analyze_match(f'../match2.csv')