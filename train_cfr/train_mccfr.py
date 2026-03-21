import json
import pickle
import random
from tqdm import tqdm

class MCCFRTrainer:
    def __init__(self):
        self.regret_sum = {}
        self.strategy_sum = {}
        
        print("📥 正在载入精确期望胜率 (EHS) 字典...")
        try:
            with open("EHS_table_fixed.pkl", "rb") as f:
                self.EHS_LOOKUP = pickle.load(f)
            with open("preflop_table.pkl", "rb") as f:
                self.PREFLOP_LOOKUP = pickle.load(f)
        except Exception as e:
            print(f"❌ 字典读取失败: {e}")
            exit()
            
        self.s0_keys = list(self.PREFLOP_LOOKUP.keys())
        self.s1_keys = [k for k in self.EHS_LOOKUP.keys() if len(k[1]) == 3] 
        self.s2_keys = [k for k in self.EHS_LOOKUP.keys() if len(k[1]) == 4] 
        print("✅ 引擎初始化完毕！完美适配 CMU gym_env.py 规则。")

    def get_infoset_key(self, player, street, ehs_val, call_amount, pot_size):
        if ehs_val >= 0.75: hs = "Nuts"
        elif ehs_val >= 0.55: hs = "Strong"
        elif ehs_val >= 0.40: hs = "Marginal"
        else: hs = "Trash"

        odds = call_amount / (pot_size + call_amount + 1e-9)
        if call_amount == 0: pressure = "None"
        elif odds < 0.20: pressure = "Low"
        elif odds < 0.35: pressure = "Med"
        else: pressure = "High"

        return f"P{player}_S{street}_{hs}_{pressure}"

    def get_strategy(self, infoset, valid_actions):
        if infoset not in self.regret_sum:
            self.regret_sum[infoset] = {a: 0.0 for a in ["F", "C", "K", "R"]}
            self.strategy_sum[infoset] = {a: 0.0 for a in ["F", "C", "K", "R"]}
            
        regrets = self.regret_sum[infoset]
        positive_regrets = {a: max(0.0, regrets.get(a, 0.0)) for a in valid_actions}
        sum_pos = sum(positive_regrets.values())
        
        strategy = {}
        for a in valid_actions:
            strategy[a] = positive_regrets[a] / sum_pos if sum_pos > 0 else 1.0 / len(valid_actions)
        return strategy

    def sample_ehs(self, street):
        if street == 0: return self.PREFLOP_LOOKUP[random.choice(self.s0_keys)]
        elif street == 1: 
            res = self.EHS_LOOKUP[random.choice(self.s1_keys)]
            return res[0] + (1 - res[0]) * res[1] - res[0] * res[2]
        else:
            keys = self.s2_keys if len(self.s2_keys) > 0 else self.s1_keys
            res = self.EHS_LOOKUP[random.choice(keys)]
            return res[0] + (1 - res[0]) * res[1] - res[0] * res[2]

    def cfr_recursive(self, p0_bet, p1_bet, street, acting_player, p0_ehs, p1_ehs, p0_prob, p1_prob, min_raise, raises_this_street):
        """
        状态机完美镜像 gym_env.py 的 step() 逻辑
        """
        pot = p0_bet + p1_bet

        # 1. Showdown (摊牌结算)
        if street > 3:
            if p0_ehs > p1_ehs: return [p1_bet, -p1_bet]
            elif p0_ehs < p1_ehs: return [-p0_bet, p0_bet]
            else: return [0.0, 0.0]

        # 2. 获取合法动作 (严格遵循引擎 _get_valid_actions)
        valid_actions = ["F"] # Fold 永远合法
        if p0_bet == p1_bet:
            valid_actions.append("K") # Bets equal -> Check (K)
        else:
            valid_actions.append("C") # Bets unequal -> Call (C)
            
        # Raise 合法性判定
        if max(p0_bet, p1_bet) < 100 and raises_this_street < 2: # 解锁 3-Bet，每街最多 2 次加注
            valid_actions.append("R")

        call_amt = p1_bet - p0_bet if acting_player == 0 else p0_bet - p1_bet
        my_ehs = p0_ehs if acting_player == 0 else p1_ehs
        infoset = self.get_infoset_key(acting_player, street, my_ehs, max(0, call_amt), pot)
        strategy = self.get_strategy(infoset, valid_actions)

        node_util = [0.0, 0.0]
        action_utils = {}

        # 3. 遍历动作并穿越博弈树
        for action in valid_actions:
            n_p0, n_p1 = p0_bet, p1_bet
            n_street = street
            n_actor = 1 - acting_player
            n_min_raise = min_raise
            n_raises = raises_this_street
            new_street_flag = False

            if action == "F":
                # 弃牌直接结算
                util = [p1_bet, -p1_bet] if acting_player == 0 else [-p0_bet, p0_bet]
                action_utils[action] = util
                node_util[0] += strategy[action] * util[0]
                node_util[1] += strategy[action] * util[1]
                continue
                
            elif action == "C":
                # 跟注匹配筹码
                if acting_player == 0: n_p0 = n_p1
                else: n_p1 = n_p0
                
                # 引擎 274 行特判：翻牌前 SB 补齐盲注，大盲继续行动
                if street == 0 and acting_player == 0 and n_p0 == 2:
                    new_street_flag = False
                else:
                    new_street_flag = True
                    
            elif action == "K":
                # 引擎 280 行特判：最后行动者 Check，街结束
                if street == 0 and acting_player == 1: new_street_flag = True
                elif street >= 1 and acting_player == 0: new_street_flag = True
                
            elif action == "R":
                # 加注逻辑 (取 20 和 引擎 min_raise 中的最大值，但不超过 100)
                raise_amt = max(20, min_raise)
                max_allowed_raise = 100 - max(p0_bet, p1_bet)
                actual_raise = min(raise_amt, max_allowed_raise)
                
                if acting_player == 0: n_p0 = n_p1 + actual_raise
                else: n_p1 = n_p0 + actual_raise
                
                n_min_raise = actual_raise
                n_raises += 1

            # 4. 递归执行下一状态
            if new_street_flag:
                n_street = street + 1
                n_actor = 1 # 引擎 229 行：翻牌后大盲(1)先行动
                n_min_raise = 2 # 引擎 225 行：新的一街最小加注恢复为大盲额度
                n_p0_ehs = self.sample_ehs(n_street)
                n_p1_ehs = self.sample_ehs(n_street)
                
                util = self.cfr_recursive(n_p0, n_p1, n_street, n_actor, n_p0_ehs, n_p1_ehs, 
                                          p0_prob * strategy.get(action, 0) if acting_player == 0 else p0_prob, 
                                          p1_prob * strategy.get(action, 0) if acting_player == 1 else p1_prob, 
                                          n_min_raise, 0)
            else:
                util = self.cfr_recursive(n_p0, n_p1, n_street, n_actor, p0_ehs, p1_ehs, 
                                          p0_prob * strategy.get(action, 0) if acting_player == 0 else p0_prob, 
                                          p1_prob * strategy.get(action, 0) if acting_player == 1 else p1_prob, 
                                          n_min_raise, n_raises)
                                          
            action_utils[action] = util
            node_util[0] += strategy[action] * util[0]
            node_util[1] += strategy[action] * util[1]

        # 5. 更新后悔值与策略
        for action in valid_actions:
            if acting_player == 0:
                regret = action_utils[action][0] - node_util[0]
                self.regret_sum[infoset][action] += p1_prob * regret
                self.strategy_sum[infoset][action] += p0_prob * strategy[action]
            else:
                regret = action_utils[action][1] - node_util[1]
                self.regret_sum[infoset][action] += p0_prob * regret
                self.strategy_sum[infoset][action] += p1_prob * strategy[action]

        return node_util

    def train(self, iterations):
        print(f"🔥 开始进行 {iterations} 次 100% 仿真博弈树穿越...")
        for _ in tqdm(range(iterations), desc="MCCFR Training"):
            # 引擎 198 行: 初始化 Pre-flop 状态 (SB=1, BB=2)
            p0_s0 = self.sample_ehs(0)
            p1_s0 = self.sample_ehs(0)
            self.cfr_recursive(p0_bet=1, p1_bet=2, street=0, acting_player=0, 
                               p0_ehs=p0_s0, p1_ehs=p1_s0, 
                               p0_prob=1.0, p1_prob=1.0, 
                               min_raise=2, raises_this_street=0)

    def export_strategy(self, filename="cfr_strategy.json"):
        final = {}
        # 映射回比赛引擎能懂的词汇：把我们的 K(Check) 和 C(Call) 分开映射
        mapping = {"F": "FOLD", "C": "CALL", "K": "CHECK", "R": "RAISE"}
        for k, v in self.strategy_sum.items():
            s = sum(v.values())
            if s > 0:
                final[k] = {mapping[a]: round(v[a]/s, 3) for a in v if v[a] > 0}
            else:
                # 兜底：没探索过的节点默认 CHECK 或 CALL
                final[k] = {"CHECK": 1.0, "CALL": 1.0}
                
        with open(filename, 'w') as f: 
            json.dump(final, f, indent=4)
        print(f"🎉 纳什均衡策略已导出至: {filename}")

if __name__ == "__main__":
    trainer = MCCFRTrainer()
    trainer.train(10000000) 
    trainer.export_strategy("cfr_strategy.json")