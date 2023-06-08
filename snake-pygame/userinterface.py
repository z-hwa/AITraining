from agent import train
from humangame import runner
from agentPlaying import agent_play
import pygame
import os

if __name__ == '__main__':

    print('////////////////////\n////////////////////\nThis is ai project for snake game')
    while True:
        # 選擇模式
        mode = int(input('which mode do you want to start with(1. player/2. agent play/3. ai training/4. exit)'))

        # 模式1: 玩家手動
        if mode == 1:
            print('In game surface, press any button to start game!')
            while True:
                runner()  # 執行者，並創建遊戲
                choose = input('Is you want to play again?(y/n)')  # 確認是否要繼續玩
                if choose == 'y':
                    continue
                else:
                    break
        elif mode == 2:
            # 模式2: ai遊玩
            while True:
                model_path = input('please input path of model which you want to use:')
                agent_play(model_path)
                choose = input('Is you want to watch agent play again?(y/n)')  # 確認是否要繼續玩
                if choose == 'y':
                    continue
                else:
                    break
        elif mode == 3:
            # 模式3: ai訓練
            print('If you want to end training, please click X in game surface.')
            print('Training model and figure will be saved in model and figure folder, respectively.')
            while True:
                file_name = input('please input model name which will be save in ./model after training:')
                train(file_name)
                choose = input('Is you want to training again?(y/n)')  # 確認是否要繼續玩
                if choose == 'y':
                    continue
                else:
                    break
        else:
            print('thank for using')
            break
