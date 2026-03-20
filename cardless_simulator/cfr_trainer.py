import json
import random

# 使用单字母，避免字符串合并时取值错误
ACTIONS = ["F", "C", "R"] 

class CFRTrainer:
    def __init__(self):
        self.regret_sum = {}
        self.strategy_sum = {}

    def get_strategy(self, infoset_key, valid_actions):
        """只针对当前合法的动作分配概率"""
        if infoset_key not in self.regret_sum:
            self.regret_sum[infoset_key] = {a: 0.0 for a in ACTIONS}
            self.strategy_sum[infoset_key] = {a: 0.0 for a in ACTIONS}
            
        regrets = self.regret_sum[infoset_key]
        positive_regrets = {a: max(0.0, regrets[a]) for a in valid_actions}
        sum_positive_regrets = sum(positive_regrets.values())

        strategy = {}
        for a in valid_actions:
            if sum_positive_regrets > 0:
                strategy[a] = positive_regrets[a] / sum_positive_regrets
            else:
                strategy[a] = 1.0 / len(valid_actions)
        return strategy

    def get_infoset_key(self, street, ehs_val, call_amount, pot_size):
        if ehs_val >= 0.75: hs_bucket = "Nuts"
        elif ehs_val >= 0.60: hs_bucket = "Strong"
        elif ehs_val >= 0.40: hs_bucket = "Marginal"
        else: hs_bucket = "Trash"

        if call_amount == 0: pressure = "None"
        else:
            pot_odds = call_amount / (pot_size + call_amount + 1e-9)
            if pot_odds < 0.20: pressure = "Low"
            elif pot_odds < 0.35: pressure = "Med"
            else: pressure = "High"

        return f"S{street}_{hs_bucket}_{pressure}"

    def cfr_recursive(self, history, p0_ehs, p1_ehs, p0_bet, p1_bet, street, p0_prob, p1_prob):
        pot = p0_bet + p1_bet
        player = len(history) % 2 
        opp = 1 - player

        # ==========================================
        # 1. 终止节点结算 (Terminal Utilities)
        # ==========================================
        if len(history) > 0:
            last_action = history[-1]
            
            # 结局 A：刚行动的对手 Fold 了，我赢走对手投入的筹码
            if last_action == "F":
                # 当前视角的 player 没做动作，是 opp 做了 F。
                # 所以 player 赢了 opp_bet。
                return p1_bet if player == 0 else p0_bet
                
            # 结局 B：对手 Call (或 Check 打平)
            if last_action == "C":
                if street == 3: # 河牌圈结束，摊牌！
                    if player == 0:
                        if p0_ehs > p1_ehs: return p1_bet
                        elif p0_ehs < p1_ehs: return -p0_bet
                        else: return 0.0
                    else:
                        if p1_ehs > p0_ehs: return p0_bet
                        elif p1_ehs < p0_ehs: return -p1_bet
                        else: return 0.0
                else: 
                    # 还没到最后，进入下一条街！
                    # 下一条街总是从 P0 (小盲注位置) 开始行动，所以 history 清空
                    next_util = self.cfr_recursive("", p0_ehs, p1_ehs, p0_bet, p1_bet, street + 1, p0_prob, p1_prob)
                    # 极其重要的博弈论数学：如果当前是 P0，下一条街也是 P0 开始，视角一致，不用加负号！
                    return next_util if player == 0 else -next_util

        # ==========================================
        # 2. 动态生成合法动作 (强制防守：筹码封顶限制)
        # ==========================================
        valid_actions = ["F", "C"]
        
        amount_to_call = p1_bet - p0_bet if player == 0 else p0_bet - p1_bet
        amount_to_call = max(0, amount_to_call)
        
        # 简化加注模型：当前落后者补齐筹码后，再多下注 20 筹码
        # 比赛最大单人下注是 100，超了就不准再 Raise，只能 Call 或 Fold
        next_raise_bet = max(p0_bet, p1_bet) + 20
        if next_raise_bet <= 100:
            valid_actions.append("R")

        # ==========================================
        # 3. 查表与向下推演
        # ==========================================
        my_ehs = p0_ehs if player == 0 else p1_ehs
        infoset = self.get_infoset_key(street, my_ehs, amount_to_call, pot)
        strategy = self.get_strategy(infoset, valid_actions)

        util = {}
        node_util = 0.0

        for action in valid_actions:
            next_history = history + action
            next_p0_bet = p0_bet
            next_p1_bet = p1_bet

            if action == "C":
                if player == 0: next_p0_bet = max(p0_bet, p1_bet)
                else: next_p1_bet = max(p0_bet, p1_bet)
            elif action == "R":
                if player == 0: next_p0_bet = next_raise_bet
                else: next_p1_bet = next_raise_bet

            # 核心：递归调用自身！因为是零和博弈，对手的收益就是我的负收益，所以加负号
            if player == 0:
                util[action] = -self.cfr_recursive(next_history, p0_ehs, p1_ehs, next_p0_bet, next_p1_bet, street, p0_prob * strategy[action], p1_prob)
            else:
                util[action] = -self.cfr_recursive(next_history, p0_ehs, p1_ehs, next_p0_bet, next_p1_bet, street, p0_prob, p1_prob * strategy[action])
            
            # 计算当前节点的平均期望收益
            node_util += strategy[action] * util[action]

        # ==========================================
        # 4. 更新遗憾值和策略表
        # ==========================================
        for action in valid_actions:
            regret = util[action] - node_util
            # 乘以对手到达这里的概率，更新遗憾值
            prob_weight = p1_prob if player == 0 else p0_prob
            self.regret_sum[infoset][action] += prob_weight * regret
            
            # 累加策略，用于最后导出最终的平均策略
            my_prob_weight = p0_prob if player == 0 else p1_prob
            self.strategy_sum[infoset][action] += my_prob_weight * strategy[action]

        return node_util

    def train(self, iterations):
        print(f"开始离线训练，迭代 {iterations} 次...")
        for i in range(iterations):
            # 每次随机给双方分配一个“底牌胜率” (0 到 1 之间)
            # 真实情况下胜率是负相关的，这里为了极简处理采用独立随机抽样
            p0_ehs = random.random()
            p1_ehs = random.random()
            
            # P0 放小盲(1)，P1 放大盲(2)，Street 从 1(Flop) 开始
            self.cfr_recursive("", p0_ehs, p1_ehs, 1, 2, 1, 1.0, 1.0)
            
            if i % 10000 == 0 and i > 0:
                print(f"已完成 {i} 次迭代...")

    def export_strategy(self, filename="cfr_strategy.json"):
        final_strategy = {}
        for infoset, strategy in self.strategy_sum.items():
            total = sum(strategy.values())
            # 将缩写转换回可读的全拼
            mapping = {"F": "FOLD", "C": "CALL", "R": "RAISE"}
            if total > 0:
                final_strategy[infoset] = {mapping[a]: round(strategy[a] / total, 3) for a in ACTIONS}
            else:
                final_strategy[infoset] = {mapping[a]: round(1.0 / len(ACTIONS), 3) for a in ACTIONS}
                
        with open(filename, 'w') as f:
            json.dump(final_strategy, f, indent=4, sort_keys=True)
        print(f"🎉 策略已导出至 {filename}！")

if __name__ == "__main__":
    trainer = CFRTrainer()
    trainer.train(5000000) # 迭代次数可以根据需要调整，越多越接近纳什均衡，建议至少10万次以上
    trainer.export_strategy()