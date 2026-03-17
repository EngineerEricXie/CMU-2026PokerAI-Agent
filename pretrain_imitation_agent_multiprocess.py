"""
Imitation Learning Pretraining
================================
讓 ActorCriticNetwork 模仿 ProbabilityAgent 的決策，
產生一個「會打基礎撲克」的初始 weights，再交給 PPO 繼續訓練。

使用方式：
    python pretrain_imitation.py

產出：
    agents/ppo_agent_weights.pth   ← 直接給 train_ppo_agent.py 用
    
"""

# 範例
# 用 fast teacher（預設，最快）
# python pretrain_imitation_agent_multiprocess.py

# # 用 ProbabilityAgent 當老師
# python pretrain_imitation_agent_multiprocess.py --teacher probability --episodes 5000

# # 用 V10 當老師（最強但最慢）
# python pretrain_imitation_agent_multiprocess.py --teacher v10_old --episodes 3000 --epochs 30

# # 指定輸出路徑（例如存成另一個檔案）
# python pretrain_imitation_agent_multiprocess.py --teacher v10 --output agents/ppo_agent_weights_v10_bk.pth
# python3 pretrain_imitation_agent_multiprocess.py --teacher v10_fat --output agents/ppo_agent_weights_v10fat_bk.pth

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gym_env import PokerEnv
from agents.prob_agent import ProbabilityAgent
from agents.v10_old import V10  # 【新增】引入 V10 老師
from agents.v10_old_fat import V10_Fat  # 【新增】引入 V10 老師
import logging                  # 【新增】用來抑制老師的 log
import concurrent.futures
import multiprocessing as mp

def fast_teacher_act(obs, evaluator):
    """
    ProbabilityAgent 策略的快速版本：
    用 30 次模擬（而非 400 次），速度快 13 倍，策略品質足夠用於預訓練。
    """
    my_cards_raw = [c for c in obs["my_cards"] if c != -1]
    community_cards = [c for c in obs["community_cards"] if c != -1]
    opp_discarded = [c for c in obs.get("opp_discarded_cards", [-1,-1,-1]) if c != -1]
    valid_actions = obs["valid_actions"]

    def mc_equity(hole_cards, sims=30):
        shown = set(hole_cards) | set(community_cards) | set(opp_discarded)
        deck  = [c for c in range(27) if c not in shown]
        wins, valid = 0, 0
        board_needed = 5 - len(community_cards)
        for _ in range(sims):
            need = 2 + board_needed
            if need > len(deck):
                continue
            sample   = random.sample(deck, need)
            opp_hand = sample[:2]
            board    = community_cards + sample[2:]
            if len(board) != 5:
                continue
            my_r  = evaluator.evaluate([PokerEnv.int_to_card(c) for c in hole_cards],
                                       [PokerEnv.int_to_card(c) for c in board])
            opp_r = evaluator.evaluate([PokerEnv.int_to_card(c) for c in opp_hand],
                                       [PokerEnv.int_to_card(c) for c in board])
            if my_r < opp_r:
                wins += 1
            valid += 1
        return wins / valid if valid > 0 else 0.5

    # Discard：選 equity 最高的兩張牌保留
    if valid_actions[PokerEnv.ActionType.DISCARD.value]:
        my_cards = my_cards_raw[:5]
        best_eq, best_keep = -1, (0, 1)
        for i in range(5):
            for j in range(i+1, 5):
                eq = mc_equity([my_cards[i], my_cards[j]])
                if eq > best_eq:
                    best_eq, best_keep = eq, (i, j)
        return (PokerEnv.ActionType.DISCARD.value, 0, best_keep[0], best_keep[1])

    # Betting：equity vs pot odds
    my_cards = my_cards_raw[:2]
    equity   = mc_equity(my_cards)
    cont_cost = obs["opp_bet"] - obs["my_bet"]
    pot_size  = obs["my_bet"] + obs["opp_bet"]
    pot_odds  = cont_cost / (cont_cost + pot_size) if cont_cost > 0 else 0

    if equity > 0.75 and valid_actions[PokerEnv.ActionType.RAISE.value]:
        amt = max(obs["min_raise"], min(obs["max_raise"], int(pot_size * 0.75)))
        return (PokerEnv.ActionType.RAISE.value, amt, 0, 0)
    elif equity >= pot_odds and valid_actions[PokerEnv.ActionType.CALL.value]:
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
    elif valid_actions[PokerEnv.ActionType.CHECK.value]:
        return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
    else:
        return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)

# ── 與 train_ppo_agent.py 完全相同的常數與函式 ──────────────────────────

PREFLOP_TABLE = {}
if os.path.exists("preflop_equity.json"):
    with open("preflop_equity.json", "r") as f:
        PREFLOP_TABLE = json.load(f)

# 全域只建一次 evaluator，避免每次 Monte Carlo 都 new PokerEnv()
_EVALUATOR = PokerEnv().evaluator

def compute_equity(obs, num_simulations=30):
    my_cards_raw = [c for c in obs["my_cards"] if c != -1]
    my_cards = my_cards_raw[:2] if len(my_cards_raw) >= 2 else my_cards_raw
    if len(my_cards) != 2:
        return 0.5
    community_cards = [c for c in obs["community_cards"] if c != -1]
    if len(community_cards) == 0 and len(my_cards) == 2:
        key = f"{min(my_cards)}_{max(my_cards)}"
        if key in PREFLOP_TABLE:
            return PREFLOP_TABLE[key]
    opp_discarded = list(obs.get("opp_discarded_cards", [-1, -1, -1]))
    shown_cards = set(my_cards) | set(community_cards) | {c for c in opp_discarded if c != -1}
    deck = list(range(27))
    non_shown_cards = [card for card in deck if card not in shown_cards]
    wins = 0
    for _ in range(num_simulations):
        opp_needed = 2
        board_needed = 5 - len(community_cards)
        sample_size = opp_needed + board_needed
        if sample_size > len(non_shown_cards):
            continue
        sample = random.sample(non_shown_cards, sample_size)
        opp_full = sample[:opp_needed]
        community_full = community_cards + sample[opp_needed:]
        my_hand   = [PokerEnv.int_to_card(c) for c in my_cards]
        opp_hand  = [PokerEnv.int_to_card(c) for c in opp_full]
        board     = [PokerEnv.int_to_card(c) for c in community_full]
        my_rank   = _EVALUATOR.evaluate(my_hand, board)
        opp_rank  = _EVALUATOR.evaluate(opp_hand, board)
        if my_rank < opp_rank:
            wins += 1
    return wins / num_simulations if num_simulations > 0 else 0.0

INPUT_DIM = 16

def preprocess_observation(obs):
    street          = np.array([obs["street"] / 3.0])
    my_cards        = np.array([((c + 1) / 28.0) if c != -1 else 0.0 for c in obs["my_cards"]])
    if len(my_cards) < 5:
        my_cards = np.resize(my_cards, 5)
    community_cards = np.array([((c + 1) / 28.0) if c != -1 else 0.0 for c in obs["community_cards"]])
    my_bet          = np.array([obs["my_bet"]   / 100.0])
    opp_bet         = np.array([obs["opp_bet"]  / 100.0])
    min_raise       = np.array([obs["min_raise"]/ 100.0])
    max_raise       = np.array([obs["max_raise"]/ 100.0])
    equity          = np.array([compute_equity(obs)])
    features = np.concatenate([street, my_cards[:5], community_cards[:5],
                                my_bet, opp_bet, min_raise, max_raise, equity])
    return torch.tensor(features, dtype=torch.float32)

KEEP_PAIRS         = [(i, j) for i in range(5) for j in range(i + 1, 5)]
KEEP_PAIR_TO_IDX   = {pair: idx for idx, pair in enumerate(KEEP_PAIRS)}
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)   # 10

# ── 與 train_ppo_agent.py 完全相同的網路架構 ────────────────────────────

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,
                 num_action_types=5, num_raise_classes=100,
                 num_discard_classes=NUM_DISCARD_CLASSES):
        super().__init__()
        self.fc1             = nn.Linear(input_dim, hidden_dim)
        self.fc2             = nn.Linear(hidden_dim, hidden_dim)
        self.action_type_head = nn.Linear(hidden_dim, num_action_types)
        self.raise_head       = nn.Linear(hidden_dim, num_raise_classes)
        self.discard_head     = nn.Linear(hidden_dim, num_discard_classes)
        self.value_head       = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return (self.action_type_head(x),
                self.raise_head(x),
                self.discard_head(x),
                self.value_head(x))

# ── 資料收集 ─────────────────────────────────────────────────────────────
# ── 資料收集 (多進程平行版) ─────────────────────────────────────────────

def collect_worker(worker_id, episodes_to_play, teacher):
    """
    獨立進程的 Worker：初始化自己的環境與老師，負責收集局數並回傳
    """
    env = PokerEnv()
    
    # 每個 Worker 必須獨立初始化自己的老師，避免進程間互相干擾
    if teacher == "probability":
        teacher_agent_0 = ProbabilityAgent()
        teacher_agent_1 = ProbabilityAgent()
    elif teacher == "v10":
        teacher_agent_0 = V10()
        teacher_agent_1 = V10()
    elif teacher == "v10_fat":
        teacher_agent_0 = V10_Fat()
        teacher_agent_1 = V10_Fat()
    else:
        teacher_agent_0 = None
        teacher_agent_1 = None

    # 關閉老師的 log 避免洗畫面
    if teacher in ["probability", "v10", "v10_fat"]:
        for t_agent in [teacher_agent_0, teacher_agent_1]:
            if hasattr(t_agent, 'logger'):
                t_agent.logger.setLevel(logging.WARNING)
                for handler in t_agent.logger.handlers:
                    handler.setLevel(logging.WARNING)

    states        = []
    action_types  = []
    raise_idxs    = []
    discard_idxs  = []

    for ep in range(episodes_to_play):
        obs, info = env.reset()
        done = False

        while not done:
            acting = obs[0]["acting_agent"]

            if acting == 0:
                teacher_obs = obs[0]
                if teacher == "fast":
                    teacher_act = fast_teacher_act(teacher_obs, _EVALUATOR)
                else:
                    teacher_act = teacher_agent_0.act(teacher_obs, reward=0, terminated=False, truncated=False, info={})
            else:
                teacher_obs = obs[1]
                if teacher == "fast":
                    teacher_act = fast_teacher_act(teacher_obs, _EVALUATOR)
                else:
                    teacher_act = teacher_agent_1.act(teacher_obs, reward=0, terminated=False, truncated=False, info={})

            state = preprocess_observation(teacher_obs).numpy()
            act_type, raise_amt, keep1, keep2 = teacher_act

            raise_idx = 0
            if act_type == PokerEnv.ActionType.RAISE.value and raise_amt > 0:
                raise_idx = max(0, min(99, raise_amt - 1))

            discard_idx = 0
            if act_type == PokerEnv.ActionType.DISCARD.value:
                pair = (min(keep1, keep2), max(keep1, keep2))
                discard_idx = KEEP_PAIR_TO_IDX.get(pair, 0)

            states.append(state)
            action_types.append(act_type)
            raise_idxs.append(raise_idx)
            discard_idxs.append(discard_idx)

            obs, reward, done, truncated, info = env.step(teacher_act)

    return states, action_types, raise_idxs, discard_idxs


def collect_demonstrations(num_episodes=20000, teacher="fast"):
    """
    派發任務給多個 CPU 核心，加速收集過程
    """
    num_workers = max(1, mp.cpu_count() - 1)
    episodes_per_worker = max(1, num_episodes // num_workers)
    actual_episodes = episodes_per_worker * num_workers

    print(f"Collecting {actual_episodes} episodes using '{teacher}' teacher...")
    print(f"Using {num_workers} parallel CPU workers...")

    all_states        = []
    all_action_types  = []
    all_raise_idxs    = []
    all_discard_idxs  = []

    # 使用 spawn 確保 PyTorch 與多進程的相容性
    ctx = mp.get_context('spawn')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = []
        for i in range(num_workers):
            futures.append(
                executor.submit(collect_worker, i, episodes_per_worker, teacher)
            )
        
        # 收集所有 Worker 的結果
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            s, a, r, d = future.result()
            all_states.extend(s)
            all_action_types.extend(a)
            all_raise_idxs.extend(r)
            all_discard_idxs.extend(d)
            
            completed += 1
            print(f"  Worker {completed}/{num_workers} finished. Gathered {len(all_states)} transitions so far.")

    print(f"Collection done: {len(all_states)} transitions from {actual_episodes} episodes")

    # 統計動作分佈
    from collections import Counter
    cnt = Counter(all_action_types)
    names = {0:"FOLD",1:"RAISE",2:"CHECK",3:"CALL",4:"DISCARD"}
    total = len(all_action_types)
    dist = " | ".join(f"{names[k]}:{cnt[k]/total*100:.1f}%" for k in sorted(names))
    print(f"Teacher action distribution: {dist}")

    # 轉為 PyTorch Tensor
    states_np = np.array(all_states, dtype=np.float32)
    return (torch.from_numpy(states_np),
            torch.tensor(all_action_types, dtype=torch.long),
            torch.tensor(all_raise_idxs,   dtype=torch.long),
            torch.tensor(all_discard_idxs, dtype=torch.long))

# ── 預訓練 ────────────────────────────────────────────────────────────────

def pretrain(num_episodes=20000, epochs=30, batch_size=512,
             lr=1e-3, weight_path=None, teacher="fast"):
    """
    teacher 選項：
      "fast"        - 內建快速版（推薦，速度快 13 倍）
      "probability" - ProbabilityAgent（品質較高，較慢）
      "v10"         - V10（最強老師，最慢）
    """
    if weight_path is None:
        weight_path = os.path.join(os.path.dirname(__file__),
                                   "agents", "ppo_agent_weights.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Teacher: {teacher}")

    # 1. 收集資料
    states, action_types, raise_idxs, discard_idxs = \
        collect_demonstrations(num_episodes, teacher=teacher)

    dataset    = TensorDataset(states, action_types, raise_idxs, discard_idxs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 建立網路
    net = ActorCriticNetwork(INPUT_DIM).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 損失函式
    # action_type：CrossEntropy（主要損失）
    # raise：只在 RAISE 時算
    # discard：只在 DISCARD 時算
    ce_loss = nn.CrossEntropyLoss()

    print(f"\nStarting pretraining: {epochs} epochs, batch_size={batch_size}")
    print("-" * 60)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss      = 0.0
        total_act_loss  = 0.0
        total_raise_loss = 0.0
        total_disc_loss  = 0.0
        n_batches = 0

        for s, at, ri, di in dataloader:
            s  = s.to(device)
            at = at.to(device)
            ri = ri.to(device)
            di = di.to(device)

            act_logits, raise_logits, discard_logits, _ = net(s)

            # Action type loss（全部樣本）
            loss_act = ce_loss(act_logits, at)

            # Raise loss（只算 RAISE 樣本）
            raise_mask = (at == PokerEnv.ActionType.RAISE.value)
            if raise_mask.sum() > 0:
                loss_raise = ce_loss(raise_logits[raise_mask], ri[raise_mask])
            else:
                loss_raise = torch.tensor(0.0, device=device)

            # Discard loss（只算 DISCARD 樣本）
            discard_mask = (at == PokerEnv.ActionType.DISCARD.value)
            if discard_mask.sum() > 0:
                loss_discard = ce_loss(discard_logits[discard_mask], di[discard_mask])
            else:
                loss_discard = torch.tensor(0.0, device=device)

            # 加權合併：action_type 最重要，raise/discard 次之
            loss = loss_act + 0.5 * loss_raise + 0.5 * loss_discard

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss       += loss.item()
            total_act_loss   += loss_act.item()
            total_raise_loss += loss_raise.item()
            total_disc_loss  += loss_discard.item()
            n_batches += 1

        avg_loss      = total_loss      / n_batches
        avg_act_loss  = total_act_loss  / n_batches
        avg_raise_loss= total_raise_loss/ n_batches
        avg_disc_loss = total_disc_loss / n_batches

        # 計算 action type 準確率（分批，避免 GPU OOM）
        net.eval()
        correct = 0
        with torch.no_grad():
            for s_batch, at_batch, _, _ in dataloader:
                s_batch  = s_batch.to(device)
                at_batch = at_batch.to(device)
                logits, _, _, _ = net(s_batch)
                correct += (logits.argmax(dim=-1) == at_batch).sum().item()
        acc = correct / len(states) * 100

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss={avg_loss:.4f} "
              f"(act={avg_act_loss:.4f} raise={avg_raise_loss:.4f} disc={avg_disc_loss:.4f}) | "
              f"ActionType Acc={acc:.1f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), weight_path)

    print(f"\nPretraining complete. Best loss={best_loss:.4f}")
    print(f"Weights saved to: {weight_path}")
    print()
    print("Next step: run train_ppo_agent.py")
    print("It will load these weights automatically if weight_path is the same.")

# ── 入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # 【重要】多進程防呆
    import argparse
    parser = argparse.ArgumentParser(description="Imitation Learning Pretraining")
    parser.add_argument("--teacher", type=str, default="fast",
                        choices=["fast", "probability", "v10", "v10_fat"],
                        help="Teacher agent: fast / probability / v10 / v10_fat (default: fast)")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Episodes to collect (default: 10000)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output weight path (default: agents/ppo_agent_weights.pth)")
    args = parser.parse_args()

    print(f"=== Imitation Learning Pretraining ===")
    print(f"Teacher:  {args.teacher}")
    print(f"Episodes: {args.episodes}")
    print(f"Epochs:   {args.epochs}")
    print()

    pretrain(
        num_episodes=args.episodes,
        epochs=args.epochs,
        batch_size=512,
        lr=1e-3,
        teacher=args.teacher,
        weight_path=args.output,
    )