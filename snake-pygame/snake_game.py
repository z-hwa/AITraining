import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()  # 初始化pygame
font = pygame.font.Font('arial.ttf', 25)  # 設定遊戲字體：arial.ttf


# 設定方向的Enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# RGB色彩
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# 設定，單位方塊的大小以及遊戲速度
BLOCK_SIZE = 20
SPEED = 20


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w  # 設定遊戲的寬度
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))  # 設定pygame的視窗長寬
        pygame.display.set_caption('Snake')  # 標題名為Snake
        self.clock = pygame.time.Clock()  # 設定遊戲時間
        self.reset()    # 重置遊戲

    def reset(self):
        self.direction = Direction.RIGHT  # 初始遊戲狀態，蛇的方向為右邊
        self.head = Point(self.w / 2, self.h / 2)  # 蛇的頭部位置 畫面中心
        # 設定整條蛇的list，以及牠的兩節身體
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0  # 初始分數0
        self.food = None  # 初始食物數量0
        self._place_food()  # 呼叫放置食物的函數
        self.frame_iteration = 0   # 重置遊戲的迭代

    # 放置食物的函數
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # 隨機生成x
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # 隨機生成y
        self.food = Point(x, y)  # 設定食物的位置

        # 如果食物的位置在蛇的身體上，重新生成
        if self.food in self.snake:
            self._place_food()

    # 玩家操作
    def play_step(self):
        self.frame_iteration += 1   # 增加遊戲迭代

        # 收集玩家輸入
        for event in pygame.event.get():
            # 退出事件
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            '''
            #玩家操作需要
            # 方向鍵事件
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
            '''

        # 移動
        self._move(action)  # AI給出動作
        self.snake.insert(0, self.head)    # 插入頭部

        # 檢查遊戲狀態
        reward = 0  # 獎勵設為0
        game_over = False   # 輸掉遊戲(bool)設為否

        # 檢測碰撞與是否超時(時間根據當前的蛇身長度)
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True    # 輸掉遊戲(bool)設為是
            reward = -10   # 遊戲獎勵設為-10
            return reward, game_over, self.score

        # 放置食物或移動
        if self.head == self.food:
            self.score += 1    # 分數+1
            reward = 10    # 遊戲獎勵設為10
            self._place_food()  # 放置新食物
        else:
            self.snake.pop()   # 刪除蛇的尾巴

        # 更新UI以及遊戲時間
        self._update_ui()
        self.clock.tick(SPEED)
        # 回傳當前狀態
        return reward, game_over, self.score

    # 碰撞函數
    def _is_collision(self, pt=None):
        # 將 pt 設為蛇頭
        if pt is None:
            pt = self.head
        # 撞到邊
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 撞到自己
        if pt in self.snake[1:]:
            return True

        # 其他情況，無碰撞
        return False

    # 更新遊戲UI
    def _update_ui(self):
        self.display.fill(BLACK)    # 填充背景顏色

        # 畫蛇的方格
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # 畫食物方格
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # 分數資訊
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()   # 更新整個畫面

    # 移動函數
    def _move(self, action):
        # 蛇頭有三種選擇:[前進, 右轉, 左轉]

        # 順時針方向list
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)  # index設為現在蛇頭的方向

        # [1, 0, 0]代表前進
        # 比較action和[1, 0, 0]的list
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]   # 方向保持不變
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4    # 向右轉=順時針旋轉=順時針方向list+1
            new_dir = clock_wise[new_idx]   # 方向保持不變
        else:
            new_idx = (idx - 1) % 4  # 向左轉=逆時針旋轉=順時針方向list-1
            new_dir = clock_wise[new_idx]  # 方向保持不變

        self.direction = new_dir    # 更新方向

        # x,y設為頭部位置
        x = self.head.x
        y = self.head.y

        # 根據self.direction移動
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        # 設定頭的位置
        self.head = Point(x, y)


'''
# 使用者介面
if __name__ == '__main__':
    game = SnakeGame()

    # game loop
    while True:
        game_over, score = game.play_step()  # 激活遊戲，收集玩家操作

        # 遊戲結束，跳出迴圈
        if game_over == True:
            break

    print('Final Score', score)

    pygame.quit()
'''
