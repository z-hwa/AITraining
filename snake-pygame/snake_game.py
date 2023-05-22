import pygame
import random
from enum import Enum
from collections import namedtuple

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


class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w  # 設定遊戲的寬度
        self.h = h

        self.display = pygame.display.set_mode((self.w, self.h))  # 設定pygame的視窗長寬
        pygame.display.set_caption('Snake')  # 標題名為Snake
        self.clock = pygame.time.Clock()  # 設定遊戲時間

        # 初始化
        self.direction = Direction.RIGHT  # 初始遊戲狀態，蛇的方向為右邊

        self.head = Point(self.w / 2, self.h / 2)  # 蛇的頭部位置 畫面中心
        # 設定整條蛇的list，以及牠的兩節身體
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0  # 初始分數0
        self.food = None  # 初始食物數量0
        self._place_food()  # 呼叫放置食物的函數

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
        # 收集玩家輸入
        for event in pygame.event.get():
            # 退出事件
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
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

        # 移動
        self._move(self.direction)  # 更新頭部的方向
        self.snake.insert(0, self.head) # 插入頭部

        # 檢查遊戲狀態
        game_over = False
        # 檢測碰撞
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # 放置食物或移動
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 更新UI以及遊戲時間
        self._update_ui()
        self.clock.tick(SPEED)
        # 回傳當前狀態
        return game_over, self.score

    # 碰撞函數
    def _is_collision(self):
        # 撞到邊
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # 撞到自己
        if self.head in self.snake[1:]:
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
    def _move(self, direction):
        # x,y設為頭部位置
        x = self.head.x
        y = self.head.y

        # 根據direction移動
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        # 設定頭的位置
        self.head = Point(x, y)


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
