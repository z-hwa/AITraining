import torch
import random  # 隨機函數
import numpy as np
import pygame
from collections import deque  # 用於儲存訓練資料
from game import SnakeGameAI, Direction, Point  # 載入遊戲本體
from model import Linear_QNet, QTrainer
from helperForAgentPlay import plot, plotClose

MAX_MEMORY = 100_000  # 最大記憶儲存量10萬
BATCH_SIZE = 1000
LR = 0.001  # 學習率

class Agent:

    def __init__(self, model_path):
        self.n_games = 0  # 紀錄遊戲局數
        self.model = Linear_QNet(11, 256, 3)  # 創建原始模型(輸入、隱藏、輸出)
        self.model.load_state_dict(torch.load(model_path))  # 載入模型

    # 獲取遊戲狀態
    def get_state(self, game):
        head = game.snake[0]  # 蛇頭

        # 設定各個方向的偵測點
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # 設定蛇頭的方向在哪邊
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

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

    # 獲得動作
    def get_action(self, state):
        final_move = [0, 0, 0]  # 最終移動方向

        state0 = torch.tensor(state, dtype=torch.float)  # 轉換狀態為張量
        prediction = self.model(state0)  # 透過模型預測下一步(執行model中的forward函數)
        move = torch.argmax(prediction).item()  # 選取預測中最大的數值，作為下一步行動 Ex. [5.0, 2.4, 1.0]
        final_move[move] = 1  # 選取最終行動方案

        return final_move  # 回傳行動


# 訓練函數
def agent_play(model_path):
    plot_score = []  # 分數
    plot_mean_score = []  # 平均分數
    total_score = 0  # 總分
    agent = Agent(model_path)  # agent
    game = SnakeGameAI()  # 遊戲

    while True:
        # 獲得舊的state
        state_old = agent.get_state(game)

        # 獲得移動
        final_move = agent.get_action(state_old)

        # 執行動作以及獲得新的狀態
        reward, done, score, endall = game.play_step(final_move)
        if endall == 1:
            plotClose()
            return

        # 遊戲結束
        if done:
            game.reset()  # 重置遊戲
            agent.n_games += 1  # 增加遊戲局數

            plot_score.append(score)  # 添加分數到展示list
            total_score += score  # 計算到目前的總分
            mean_score = total_score / agent.n_games  # 計算平均
            plot_mean_score.append(mean_score)  # 添加平均分數到展示list
            plot(plot_score, plot_mean_score, model_path)  # 展示
