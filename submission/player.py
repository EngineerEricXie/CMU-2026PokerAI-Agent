import os
import torch
import numpy as np

# 匯入你定義的基礎介面與環境常數
from agent import Agent
from gym_env import PokerEnv

# 匯入我們在訓練腳本中定義的網路架構與預處理工具
# (確保這些函數與類別在 train_rl_agent.py 或獨立檔案中可以被 import)
from train_rl_agent import ActorCriticNetwork, preprocess_observation, KEEP_PAIRS, NUM_DISCARD_CLASSES

class TrainedPPOAgent(Agent):
    def __init__(self, weight_path="agents/ppo_agent_weights.pth", stream=False, player_id=None):
        # 初始化 FastAPI 伺服器與日誌
        super().__init__(stream=stream, player_id=player_id)
        
        # 1. 判斷運算設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initializing TrainedPPOAgent on {self.device}")
        
        # 2. 初始化神經網路 (16維度特徵)
        self.policy = ActorCriticNetwork(input_dim=16)
        
        # 3. 載入訓練好的權重
        if os.path.exists(weight_path):
            # 使用 map_location 確保在純 CPU 環境下也能讀取 GPU 訓練的權重
            self.policy.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.logger.info(f"Successfully loaded weights from {weight_path}")
        else:
            self.logger.warning(f"Weight file not found at {weight_path}. Using random initialized weights.")
            
        # 4. 移至運算設備並設定為評估模式
        self.policy.to(self.device)
        self.policy.eval()

    def __name__(self):
        return "TrainedPPOAgent"

    def act(self, observation, reward, terminated, truncated, info) -> tuple[int, int, int, int]:
        """
        接收當前狀態，透過訓練好的神經網路回傳決策動作。
        """
        # 1. 特徵預處理
        # 將字典格式的 observation 轉換為神經網路需要的 tensor，並增加 batch 維度 (unsqueeze(0))
        state = preprocess_observation(observation).unsqueeze(0).to(self.device)
        
        valid_actions = torch.tensor(observation["valid_actions"], dtype=torch.float32, device=self.device)
        min_raise = observation["min_raise"]
        max_raise = observation["max_raise"]

        # 2. 透過網路進行推論 (不計算梯度)
        with torch.no_grad():
            action_type_logits, raise_logits, discard_logits, _ = self.policy(state)

            # Mask 掉當前不合法的動作
            mask = (valid_actions == 0)
            masked_logits = action_type_logits.clone()
            masked_logits[mask] = -1e9

            # 3. 選擇最佳動作 (Greedy)
            # 使用 argmax 取代訓練時的 sample()，穩定選擇網路認為勝率最高的動作
            action_type = torch.argmax(masked_logits).item()
            raise_idx = torch.argmax(raise_logits).item()
            discard_idx = torch.argmax(discard_logits).item()

        # 4. 解析網路輸出，轉換為環境接受的格式
        raise_amount = raise_idx + 1
        if action_type == PokerEnv.ActionType.RAISE.value:
            # 確保加注金額在合法區間內
            raise_amount = int(max(min(raise_amount, max_raise), min_raise))
        else:
            raise_amount = 0

        if action_type == PokerEnv.ActionType.DISCARD.value:
            # 解析要保留的兩張底牌索引
            keep1, keep2 = KEEP_PAIRS[discard_idx % NUM_DISCARD_CLASSES]
        else:
            keep1, keep2 = 0, 0

        self.logger.info(f"Agent Action decided: Type={action_type}, Raise={raise_amount}, Keep=({keep1}, {keep2})")
        
        # 回傳符合 ActionResponse 定義的 Tuple
        return (action_type, raise_amount, keep1, keep2)

    def observe(self, observation, reward, terminated, truncated, info) -> None:
        """
        如果需要在對手的回合進行觀察或記錄，可以實作在這裡。
        (目前純推論狀態下，通常不需要做特別處理)
        """
        pass