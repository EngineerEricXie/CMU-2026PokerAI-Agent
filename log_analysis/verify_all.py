import pickle
import time

def verify_all_fixed_ehs():
    file_path = "EHS_table_fixed.pkl"
    print(f"🔍 開始全表深度掃描: {file_path}")
    
    start_time = time.time()
    try:
        with open(file_path, "rb") as f:
            table = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到 {file_path}，請確認你已經執行過修復腳本！")
        return
        
    print(f"  • 總資料筆數: {len(table)}")
    
    # 初始化極值容器
    min_hs, max_hs = float('inf'), float('-inf')
    min_ppot, max_ppot = float('inf'), float('-inf')
    min_npot, max_npot = float('inf'), float('-inf')
    min_ehs, max_ehs = float('inf'), float('-inf')
    
    print("⏳ 正在對 524 萬筆資料進行極限壓測...")
    
    for (H, C), (hs, ppot, npot) in table.items():
        # 1. 檢查三大指標
        if hs < min_hs: min_hs = hs
        if hs > max_hs: max_hs = hs
        if ppot < min_ppot: min_ppot = ppot
        if ppot > max_ppot: max_ppot = ppot
        if npot < min_npot: min_npot = npot
        if npot > max_npot: max_npot = npot
        
        # 2. 模擬 Agent 在比賽中的 EHS 總和計算
        ehs = hs + (1 - hs) * ppot - hs * npot
        
        if ehs < min_ehs: min_ehs = ehs
        if ehs > max_ehs: max_ehs = ehs
        
    elapsed = time.time() - start_time
    
    print("\n📊 【全表 524 萬筆資料極值掃描結果】")
    print(f"  • HS   範圍: Min = {min_hs:.5f}, Max = {max_hs:.5f}")
    print(f"  • PPot 範圍: Min = {min_ppot:.5f}, Max = {max_ppot:.5f}")
    print(f"  • NPot 範圍: Min = {min_npot:.5f}, Max = {max_npot:.5f}")
    print(f"  • EHS  總和: Min = {min_ehs:.5f}, Max = {max_ehs:.5f}")
    
    # 浮點數誤差容忍值 (1e-6)
    tolerance = 1e-6
    if (min_hs >= -tolerance and max_hs <= 1.0 + tolerance and 
        min_ppot >= -tolerance and max_ppot <= 1.0 + tolerance and 
        min_npot >= -tolerance and max_npot <= 1.0 + tolerance and
        min_ehs >= -tolerance and max_ehs <= 1.0 + tolerance):
        print("\n✅ 恭喜！全表所有機率指標均完美落在 0.0 ~ 1.0 之間！數學邏輯已徹底修復！")
    else:
        print("\n🚨 警告：仍有數值超出 0.0 ~ 1.0 範圍，請檢查邏輯。")
        
    print(f"\n⏱️ 掃描耗時: {elapsed:.2f} 秒")

if __name__ == "__main__":
    verify_all_fixed_ehs()