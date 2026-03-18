import pickle
import os

def check_preflop():
    file_path = "preflop_table.pkl"
    if not os.path.exists(file_path):
        print(f"❌ 找不到 {file_path}")
        return

    print(f"\n🔍 開始體檢: {file_path}")
    with open(file_path, "rb") as f:
        table = pickle.load(f)
        
    print(f"  • 總資料筆數: {len(table)}")
    
    # 抽樣檢查前 3 筆
    print("  • 抽樣檢查前 3 筆資料:")
    count = 0
    min_val = float('inf')
    max_val = float('-inf')
    
    for k, v in table.items():
        if count < 3:
            print(f"    Key: {k} -> Value: {v}")
            count += 1
            
        # 尋找極值 (確保是 0.0 ~ 1.0 的機率)
        if isinstance(v, (int, float)):
            if v < min_val: min_val = v
            if v > max_val: max_val = v
            
    print(f"  • 數據範圍: Min = {min_val}, Max = {max_val}")
    
    if min_val < 0 or max_val > 1.0:
        print("  🚨 [警告] 發現異常數值！Pre-flop 勝率必須在 0.0 ~ 1.0 之間！")
    else:
        print("  ✅ 數據範圍正常。")

def check_ehs():
    file_path = "EHS_table_fixed.pkl"
    if not os.path.exists(file_path):
        print(f"❌ 找不到 {file_path}")
        return

    print(f"\n🔍 開始體檢: {file_path}")
    with open(file_path, "rb") as f:
        table = pickle.load(f)
        
    print(f"  • 總資料筆數: {len(table)}")
    
    # 抽樣檢查前 3 筆
    print("  • 抽樣檢查前 3 筆資料:")
    count = 0
    
    # 記錄三大指標的極值
    min_hs, max_hs = float('inf'), float('-inf')
    min_ppot, max_ppot = float('inf'), float('-inf')
    min_npot, max_npot = float('inf'), float('-inf')
    
    for k, v in table.items():
        if count < 3:
            print(f"    Key: {k} -> Value: {v}")
            count += 1
            
        if isinstance(v, tuple) and len(v) == 3:
            hs, ppot, npot = v
            if hs < min_hs: min_hs = hs
            if hs > max_hs: max_hs = hs
            if ppot < min_ppot: min_ppot = ppot
            if ppot > max_ppot: max_ppot = ppot
            if npot < min_npot: min_npot = npot
            if npot > max_npot: max_npot = npot
            
    print(f"  • HS   數據範圍: Min = {min_hs}, Max = {max_hs}")
    print(f"  • PPot 數據範圍: Min = {min_ppot}, Max = {max_ppot}")
    print(f"  • NPot 數據範圍: Min = {min_npot}, Max = {max_npot}")
    
    if min_hs < 0 or max_hs > 1.0 or min_ppot < 0 or max_ppot > 1.0 or min_npot < 0 or max_npot > 1.0:
        print("  🚨 [警告] 發現異常數值！EHS 指標必須在 0.0 ~ 1.0 之間！")
    else:
        print("  ✅ EHS 數據範圍正常。")

if __name__ == "__main__":
    print("=========================================")
    print("🏥 AI 預算字典健康檢查中心 🏥")
    print("=========================================")
    check_preflop()
    check_ehs()
    print("=========================================")