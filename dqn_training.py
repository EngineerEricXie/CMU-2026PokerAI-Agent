import os
import time
from gym_env import PokerEnv

# 引入你的 DQN Learner
from dqn_agent import PlayerAgent as DQNAgent

# 引入你的 Fast Bot 作為對手
from agents.PreCompute_EHS_V2 import PC_EHS_V2
from agents.PreCompute_EHS_V2 import PC_EHS_V2 as FastBot
from agents.v10_training import V10

def run_marl_training(total_hands=100000, save_interval=1000, log_interval=100, learner=DQNAgent(stream=False, is_training=True, start_from_new=True), opponent=FastBot(stream=False, is_training=True)):
    env = PokerEnv()
    
    # 建立雙方選手 (將 stream 設為 False，讓終端機版面保持乾淨)
    # learner = DQNAgent(stream=False, is_training=True, start_from_new=True)
    learner.total_hands = total_hands
    
    # opponent = FastBot(stream=False, is_training=True)
    
    agents = {0: learner, 1: opponent}
    
    print(f"🚀 開始雙人對戰訓練！")
    print(f"🥊 Player 0 (Learner): {learner.__name__()} VS Player 1 (Opponent): {opponent.__name__()}")
    print(f"預計進行 {total_hands} 局。\n" + "-"*60)
    start_time = time.time()
    
    # 💰 建立雙方總籌碼計數器
    learner_bankroll = 0.0
    opponent_bankroll = 0.0
    
    for hand in range(1, total_hands + 1):
        # 1. 初始化環境：輪流當小盲注確保公平
        small_blind_player = hand % 2
        (obs0, obs1), info = env.reset(options={"small_blind_player": small_blind_player})
        info["hand_number"] = hand
        
        terminated = False
        truncated = False
        reward0 = reward1 = 0
        
        # 2. 單局遊戲迴圈
        while not (terminated or truncated):
            current_player = obs0["acting_agent"]
            
            active_agent = agents[current_player]
            current_obs = obs0 if current_player == 0 else obs1
            current_reward = reward0 if current_player == 0 else reward1
            
            # 採取行動
            action = active_agent.act(current_obs, current_reward, terminated, truncated, info)
            
            # 執行動作
            (next_obs0, next_obs1), (next_reward0, next_reward1), terminated, truncated, info = env.step(action)
            
            # 觀察結果並學習
            learner.observe(next_obs0, next_reward0, terminated, truncated, info)
            opponent.observe(next_obs1, next_reward1, terminated, truncated, info)
            
            obs0, obs1 = next_obs0, next_obs1
            reward0, reward1 = next_reward0, next_reward1
            
        # 3. 本局結束，結算總籌碼
        learner_bankroll += reward0
        opponent_bankroll += reward1
        
        # 📊 4. 週期性印出訓練戰報 (預設每 100 局印一次)
        if hand % log_interval == 0:
            # 計算勝負差
            profit_diff = learner_bankroll - opponent_bankroll
            trend = "🟢 領先" if profit_diff > 0 else "🔴 落後" if profit_diff < 0 else "⚪ 平手"
            
            print(f"📊 [Hand {hand:6d}] {trend} | "
                  f"Learner 籌碼: {learner_bankroll:8.1f} | "
                  f"Opponent 籌碼: {opponent_bankroll:8.1f} | "
                  f"Epsilon (探索率): {learner.epsilon:.3f}")
            
        # 5. 週期性存檔
        if hand % save_interval == 0:
            model_path = f"models/dqn_poker_hand_{hand}.pth"
            os.makedirs("models", exist_ok=True)
            learner.save_model(model_path)
            print(f"💾 [Hand {hand:6d}] 已儲存模型進度，累積耗時: {(time.time() - start_time):.1f} 秒。\n" + "-"*60)
            
    learner.save_model("models/dqn_poker_final.pth")
    print("🎉 雙人對戰訓練圓滿結束！")

if __name__ == "__main__":
    # 可以自由調整 log_interval 來決定印出戰報的頻率
    # learner = DQNAgent(stream=False, is_training=True, start_from_new=True)
    # run_marl_training(total_hands=10000, save_interval=1000, log_interval=100, learner=learner, opponent=PC_EHS_V2(stream=False, is_training=True))
    learner = DQNAgent(stream=False, is_training=True, start_from_new=False, load_model_path="models/dqn_poker_final.pth")
    run_marl_training(total_hands=3000, save_interval=1000, log_interval=100, learner=learner, opponent=V10(stream=False, is_training=True))
