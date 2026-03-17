import itertools
import pickle
import time
# 請確保這裡 import 的名稱符合你遊戲引擎的檔案名稱
from gym_env import PokerEnv, WrappedEval 
from tqdm import tqdm

def build_7card_lookup_table():
    print("開始建構 7 張牌絕對牌力字典 (Key = 0~26 Integer Tuple)...")
    start_time = time.time()
    
    evaluator = WrappedEval()
    
    # 【關鍵修改】：我們直接使用 0 到 26 的整數作為迴圈基礎
    deck_ints = list(range(27))
    
    lookup_table = {}

    total_combos = len(list(itertools.combinations(deck_ints, 7)))
    
    # 窮舉所有 27 取 7 的整數組合
    for combo_ints in tqdm(itertools.combinations(deck_ints, 7), total=total_combos, desc="建構進度"):
        # 為了給 treys 算分，我們把整數轉換成 treys 能懂的卡片物件
        treys_cards = [PokerEnv.int_to_card(i) for i in combo_ints]
        
        # treys 需要分成 hand 和 board，我們隨便切 2 和 5
        hand = treys_cards[:2]
        board = treys_cards[2:]
        
        # 取得分數
        score = evaluator.evaluate(hand, board)
        
        # 【關鍵修改】：我們用「小整數的 tuple」當作字典的 Key！
        # 這樣就能完美對應你在比賽中拿到的 observation
        combo_key = tuple(sorted(combo_ints))
        lookup_table[combo_key] = score
        
    end_time = time.time()
    print(f"建構完成！共產生 {len(lookup_table)} 筆資料。")
    print(f"耗時: {end_time - start_time:.2f} 秒")
    
    return lookup_table

if __name__ == "__main__":
    table = build_7card_lookup_table()
    
    output_filename = "lookup_table_7cards.pkl"
    print(f"正在將字典存入 {output_filename} ...")
    
    with open(output_filename, "wb") as f:
        pickle.dump(table, f)
        
    print("儲存成功！請確保這個新檔案覆蓋了舊的。")