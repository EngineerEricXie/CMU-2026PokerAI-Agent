import os
import time
from gym_env import PokerEnv

# 引入你的 DQN Learner
from dqn_agent import PlayerAgent as DQNAgent

def run_marl_training(total_hands=100000, save_interval=1000, log_interval=100):
    env = PokerEnv()
    
    # 建立 Learner (負責學習，不斷更新權重)
    learner = DQNAgent(stream=False, is_training=True, start_from_new=False)
    learner.total_hands = total_hands
    
    # 建立 Opponent (不學習，扮演「過去的 Learner」)
    opponent = DQNAgent(stream=False, is_training=False, start_from_new=False)
    
    # 🌟 開局同步：確保對手擁有與 Learner 一模一樣的初始權重
    opponent.q_net.load_state_dict(learner.q_net.state_dict())
    opponent.target_net.load_state_dict(learner.target_net.state_dict())
    # 給對手保留一點點隨機性 (5%)，這樣 Learner 才能見識到不同的牌局發展
    opponent.epsilon = 0.05 
    
    agents = {0: learner, 1: opponent}
    
    print(f"🚀 開始標準【自我對弈 (Self-Play)】訓練！")
    print(f"🥊 Player 0 (Learner) VS Player 1 (Opponent Snapshot)")
    print(f"預計進行 {total_hands} 局。\n" + "-"*60)
    start_time = time.time()
    
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
            
            # 觀察結果並學習 (注意：只有 Learner 會真正觸發 Backpropagation)
            learner.observe(next_obs0, next_reward0, terminated, truncated, info)
            opponent.observe(next_obs1, next_reward1, terminated, truncated, info)
            
            obs0, obs1 = next_obs0, next_obs1
            reward0, reward1 = next_reward0, next_reward1
            
        # 3. 本局結束，結算總籌碼
        learner_bankroll += reward0
        opponent_bankroll += reward1
        
        # 4. 週期性印出訓練戰報
        if hand % log_interval == 0:
            profit_diff = learner_bankroll - opponent_bankroll
            trend = "🟢 領先" if profit_diff > 0 else "🔴 落後" if profit_diff < 0 else "⚪ 平手"
            
            print(f"📊 [Hand {hand:6d}] {trend} | "
                  f"Learner 籌碼: {learner_bankroll:8.1f} | "
                  f"Opponent 籌碼: {opponent_bankroll:8.1f} | "
                  f"Epsilon: {learner.epsilon:.3f}")
            
        # 5. 🌟 週期性存檔 與 權重同步 (Evolution!)
        if hand % save_interval == 0:
            # 儲存 Learner 目前的模型
            model_path = f"models/dqn_poker_hand_{hand}.pth"
            os.makedirs("models", exist_ok=True)
            learner.save_model(model_path)
            
            # 將 Learner 的大腦複製給 Opponent
            opponent.q_net.load_state_dict(learner.q_net.state_dict())
            opponent.target_net.load_state_dict(learner.target_net.state_dict())
            
            print(f"💾 [Hand {hand:6d}] 已存檔，並將最新權重【同步】給對手！累積耗時: {(time.time() - start_time):.1f} 秒。\n" + "-"*60)
            
    learner.save_model("models/dqn_poker_final.pth")
    print("🎉 自我對弈訓練圓滿結束！")

if __name__ == "__main__":
    run_marl_training(total_hands=100000, save_interval=1000, log_interval=100)