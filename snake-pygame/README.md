# 檔案與資料夾說明

## 手動操作
userinterface.py: 操作介面，請一律執行本程式
humangame.py: 玩家操作的遊戲本體程式

## AI訓練
model.py: AI訓練(遊玩)的模型以及算法
agent.py: AI訓練時使用的智能體
game.py: AI訓練(遊玩)使用的遊戲本體程式
helper.py: 負責顯示AI訓練過程的訓練資料圖表
helperSave.py: 負責儲存AI訓練時的訓練資料圖表以及訓練資料

## AI遊玩
agentPlaying.py: AI遊玩時使用的智能體(必須和訓練時的配置相同)
helperForAgentPlay: 負責顯示AI遊玩的資料表

## 資料
record_model_train: 模型訓練日誌
model: 存放訓練過的模型
figure: 存放已訓練模型的訓練圖表
data: 存放已訓練模型的訓練資料(更新 舊模型訓練功能以後 訓練的模型才有資料)
arial.ttf: 字型檔案
