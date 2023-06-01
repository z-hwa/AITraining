import torch
import random  # 隨機函數
import numpy as np
from collections import deque  # 用於儲存訓練資料
from game import SnakeGameAI, Direction, Point  # 載入遊戲本體
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000  # 最大記憶儲存量10萬
BATCH_SIZE = 1000
LR = 0.001  # 學習率


class Agent:

    def __init__(self):
        self.n_games = 0  # 紀錄遊戲局數
        self.epsilon = 0  # 隨機損失區間(率)
        self.gamma = 0.9    # 小於1
        ''' 
            gamma
            折扣率: 用於讓agent衡量，是要更關注長期還是短期的獎勵
            高: 關注未來
            低: 關注現在獎勵
        '''

        self.memory = deque(maxlen=MAX_MEMORY)  # deque資料結構，從左邊丟出(先入先出)，用於agent的記憶
        self.model = Linear_QNet(11, 256, 3)    # 創建原始模型(輸入、隱藏、輸出)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)    # 創建訓練者

    # 獲取遊戲狀態
    def get_state(self, game):
        head = game.snake[0]  # 蛇頭

        # 設定各個方向的偵測點
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # 設定蛇頭的方向在哪邊
        dir_l = game.direction == Direction.Left
        dir_r = game.direction == Direction.Right
        dir_u = game.direction == Direction.Up
        dir_d = game.direction == Direction.Down

        # 狀態
        state = [
            # 存在危險的動作是否為直行
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # 存在危險的動作是否為右轉
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # 存在危險的動作是否為左轉
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # 移動方向
            dir_r,
            dir_d,
            dir_u,
            dir_l,

            # 食物方位
            game.food.x < game.head.x,  # 食物在左邊
            game.food.x > game.head.x,  # 食物在右邊
            game.food.y < game.head.y,  # 食物在上邊
            game.food.y > game.head.y  # 食物在下邊
        ]

        return np.array(state, dtype=int)  # 回傳整數狀態數組

    # 記憶(狀態、動作、獎勵、下一步狀態、遊戲結束與否)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 增加記憶(tuple)(如果超過MAX_MEMORY，從左邊丟出)

    def train_long_memory(self):
        # 如果可取樣本超過BATCH_SIZE
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # (tuple)隨機獲取等同於BATCH_SIZE的樣本
        else:
            mini_sample = self.memory  # 直接拿memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # 將樣本tuple的每一個項，各自集合成一類用於訓練
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # 多組數據的訓練

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)  # 單組數據的訓練

    # 獲得動作
    def get_action(self, state):
        # 隨機移動: tradeoff exploration /  exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]  # 最終移動方向

        '''
        如果遊戲局數越大，隨機性越低:
        當遊戲局數變大 -> epsilon變低 -> 0~200隨機出現小於epsilon的可能性越低 -> 進入隨機選取步驟的可能越低
        直到局數大於等於80，不再隨機選取步數
        '''
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # 轉換狀態為張量
            prediction = self.model(state0)  # 透過模型預測下一步(執行model中的forward函數)
            move = torch.argmax(prediction).item()  # 選取預測中最大的數值，作為下一步行動 Ex. [5.0, 2.4, 1.0]
            final_move[move] = 1    # 選取最終行動方案

        return final_move   # 回傳行動


# 訓練函數
def train():
    plot_score = []  # 分數
    plot_mean_score = []  # 平均分數
    total_score = 0  # 總分
    record = 0  # 記錄
    agent = Agent()  # agent
    game = SnakeGameAI()  # 遊戲

    while True:
        # 獲得舊的state
        state_old = agent.get_state(game)

        # 獲得移動
        final_move = agent.get_action(state_old)

        # 執行動作以及獲得新的狀態
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 以short memory訓練
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 記憶
        agent.remember(state_old, final_move, reward, state_new, done)

        # 遊戲結束
        if done:
            # 以long memory訓練，並plot result(幫助agent改進自己)
            game.reset()  # 重置遊戲
            agent.n_games += 1  # 增加遊戲局數
            agent.train_long_memory()  # long memory訓練

            # 更新最高分數
            if score > record:
                record = score
                agent.model.save()  # 呼叫儲存模型的函數

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # TODO: plot


# 主程式
if __name__ == '__main__':
    train()
