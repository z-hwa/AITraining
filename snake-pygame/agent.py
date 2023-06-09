import torch
import random  # 隨機函數
import numpy as np
import os
from collections import deque  # 用於儲存訓練資料
from game import SnakeGameAI, Direction, Point  # 載入遊戲本體
from model import Linear_QNet, QTrainer
from helper import plot
from helperSave import slot, slotSave
import copy

MAX_MEMORY = 100_000  # 最大記憶儲存量5萬
BATCH_SIZE = 1000
LR = 0.001  # 學習率

UPPER_EPSILON = 80  # epsilon的上限
AI_MANIPULATE_RATE = 2.5  # 數值越高 AI操控機率越高
BODY_NUM = 0 * 2  # 身體數量(x+y)
FOOD_NUM = 0
INPUT_LAYER = 11 + BODY_NUM + FOOD_NUM  # 輸入層總數

FIGURE_RECORD_FRE = 100  # 訓練紀錄圖片 存檔頻率


class Agent:

    def __init__(self):
        self.n_games = 0  # 紀錄遊戲局數
        self.epsilon = 0  # 隨機損失區間(率)
        self.gamma = 0.9  # 小於1
        ''' 
            gamma
            折扣率: 用於讓agent衡量，是要更關注長期還是短期的獎勵
            高: 關注未來
            低: 關注現在獎勵
        '''

        self.memory = deque(maxlen=MAX_MEMORY)  # deque資料結構，從左邊丟出(先入先出)，用於agent的記憶
        self.model = Linear_QNet(INPUT_LAYER, 256, 3)  # 創建原始模型(輸入、隱藏、輸出)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # 創建訓練者

    def proceed(self, model_path, numbers_game):
        self.model.load_state_dict(torch.load(model_path + '.pth'))  # 載入模型
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # 創建訓練者
        self.n_games = numbers_game  # 載入局數

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

        food_pos = [game.food.x / 640, game.food.y / 480]
        pos = []

        for idx in range(0, int(BODY_NUM / 2)):
            if idx < len(game.snake):
                pos.append(game.snake[idx].x / 640.0)
                pos.append(game.snake[idx].y / 480.0)
            else:
                pos.append(-0.00001)
                pos.append(-0.00001)

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

        if BODY_NUM > 0:
            state.extend(pos)
        # state.extend(food_pos)

        return np.array(state, dtype=float)  # 回傳整數狀態數組

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
    def get_action(self, state, game):
        # 隨機移動: tradeoff exploration /  exploitation
        self.epsilon = UPPER_EPSILON - self.n_games  # / (len(game.snake) - 2)
        final_move = [0, 0, 0]  # 最終移動方向

        '''
        如果遊戲局數越大，隨機性越低:
        當遊戲局數變大 -> epsilon變低 -> 0~200隨機出現小於epsilon的可能性越低 -> 進入隨機選取步驟的可能越低
        直到局數大於等於80，不再隨機選取步數
        '''
        if random.randint(0, int(UPPER_EPSILON * AI_MANIPULATE_RATE)) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # 轉換狀態為張量

            # GPU
            if torch.cuda.is_available():
                state0 = state0.cuda()

            prediction = self.model(state0)  # 透過模型預測下一步(執行model中的forward函數)
            move = torch.argmax(prediction).item()  # 選取預測中最大的數值，作為下一步行動 Ex. [5.0, 2.4, 1.0]
            final_move[move] = 1  # 選取最終行動方案

        return final_move  # 回傳行動


# 訓練函數
def train(file_name):
    plot_score = []  # 分數
    plot_mean_score = []  # 平均分數
    total_score = 0  # 總分
    record = 0  # 記錄
    agent = Agent()  # agent
    game = SnakeGameAI()  # 遊戲
    record_ava = True  # 是否可以記錄該局資料

    while True:
        # 獲得舊的state
        state_old = agent.get_state(game)

        # 獲得移動
        final_move = agent.get_action(state_old, game)

        # 執行動作以及獲得新的狀態
        reward, done, score, endall = game.play_step(final_move)

        # 透過點x完全關閉訓練，儲存訓練圖
        if endall == 1:
            slotSave(file_name, plot_score, plot_mean_score, total_score, record, agent.n_games)
            return
        if agent.n_games % FIGURE_RECORD_FRE == 0 and record_ava == True:
            slotSave(file_name, plot_score, plot_mean_score, total_score, record, agent.n_games)
            record_ava = False

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
            record_ava = True  # 這局可以記錄訓練資料
            agent.train_long_memory()  # long memory訓練

            # 資料展示
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_score.append(score)  # 添加分數到展示list
            total_score += score  # 計算到目前的總分
            mean_score = total_score / agent.n_games  # 計算平均
            plot_mean_score.append(mean_score)  # 添加平均分數到展示list
            plot(plot_score, plot_mean_score, file_name)  # 展示
            slot(plot_score, plot_mean_score, file_name)  # 儲存資料

            # 更新最高分數
            if score > record:
                record = score
                agent.model.save(file_name)  # 呼叫儲存模型的函數
                slotSave(file_name, plot_score, plot_mean_score, total_score, record, agent.n_games)  # 儲存當前訓練圖


# 訓練函數
def train_proceed(file_path):
    plot_score = []  # 分數
    plot_mean_score = []  # 平均分數
    total_score = 0  # 總分
    record = 0  # 記錄
    agent = Agent()  # agent
    record_ava = True  # 是否可以記錄該局資料

    # 處理繼續訓練模型
    (file_path, file_name) = os.path.split(file_path)

    f = open('./data/' + file_name + '.txt', 'r')
    datas = f.readlines()

    list_score = datas[0].split(',')
    plot_score = [float(i) for i in list_score]

    list_mean_score = datas[1].split(',')
    plot_mean_score = [float(i) for i in list_mean_score]

    total_score = float(datas[2])
    record = float(datas[3])
    numbers_game = int(datas[4])

    file_path = os.path.join(file_path, file_name)
    agent.proceed(file_path, numbers_game)
    # 

    game = SnakeGameAI()  # 遊戲

    while True:
        # 獲得舊的state
        state_old = agent.get_state(game)

        # 獲得移動
        final_move = agent.get_action(state_old, game)

        # 執行動作以及獲得新的狀態
        reward, done, score, endall = game.play_step(final_move)

        # 透過點x完全關閉訓練，儲存訓練圖
        if endall == 1:
            slotSave(file_name, plot_score, plot_mean_score, total_score, record, agent.n_games)
            return
        if agent.n_games % FIGURE_RECORD_FRE == 0 and record_ava == True:
            slotSave(file_name, plot_score, plot_mean_score, total_score, record, agent.n_games)
            record_ava = False

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
            record_ava = True  # 這局可以記錄訓練資料
            agent.train_long_memory()  # long memory訓練

            # 展示
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_score.append(score)  # 添加分數到展示list
            total_score += score  # 計算到目前的總分
            mean_score = total_score / agent.n_games  # 計算平均
            plot_mean_score.append(mean_score)  # 添加平均分數到展示list
            plot(plot_score, plot_mean_score, file_name)  # 展示
            slot(plot_score, plot_mean_score, file_name)  # 儲存資料

            # 更新最高分數
            if score > record:
                record = score
                agent.model.save(file_name)  # 呼叫儲存模型的函數
                slotSave(file_name, plot_score, plot_mean_score, total_score, record, agent.n_games)  # 儲存當前訓練圖
