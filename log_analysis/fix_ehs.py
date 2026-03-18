import pickle
import time

def fix_ehs_table():
    print("🛠️ 開始讀取受損的 EHS 字典...")
    start_time = time.time()
    
    with open("EHS_table_final.pkl", "rb") as f:
        broken_table = pickle.load(f)
        
    fixed_table = {}
    
    print("🔧 正在進行機率分母校正...")
    for (H, C), (hs, ppot, npot) in broken_table.items():
        if len(C) == 3:
            # Flop: 除以未來的 190 種公牌組合
            fixed_ppot = ppot / 190.0
            fixed_npot = npot / 190.0
        elif len(C) == 4:
            # Turn: 除以未來的 19 種公牌組合
            fixed_ppot = ppot / 19.0
            fixed_npot = npot / 19.0
        else:
            fixed_ppot = ppot
            fixed_npot = npot
            
        fixed_table[(H, C)] = (hs, fixed_ppot, fixed_npot)
        
    with open("EHS_table_fixed.pkl", "wb") as f:
        pickle.dump(fixed_table, f)
        
    elapsed = time.time() - start_time
    print(f"✅ 修復完成！耗時 {elapsed:.2f} 秒。")
    print("💾 已產生全新完美字典: EHS_table_fixed.pkl")
    
    # 順便驗證一下修復結果
    print("\n🔍 抽樣驗證第一筆資料:")
    sample_key = list(fixed_table.keys())[0]
    print(f"Key: {sample_key} -> Value: {fixed_table[sample_key]}")

if __name__ == "__main__":
    fix_ehs_table()