import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # 會去呼叫父類別的init(nn.module)

        self.linear1 = nn.Linear(input_size, hidden_size)  # 第一個線性算子(輸入->隱藏)
        self.linear2 = nn.Linear(hidden_size, output_size)  # 第二個線性算子(隱藏->輸出)

    def forward(self, x):
        x = F.relu(self.linear1(x))  # 先應用線性層1，接著透過F.relu致動
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'  # 模型儲存路徑
        # 如果該路徑不存在
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # 創建該路徑

        file_name = os.path.join(model_folder_path, file_name)  # 設置檔案名
        torch.save(self.state_dict(), file_name)  # 儲存模型


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr  # learning rate
        self.gamma = gamma  # discount rate
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # 最佳化方法(Adam)
        self.criterion = nn.MSELoss()  # 損失算法(MSE)

    def train_step(self, state, action, reward, next_state, done):
        # 透過torch.tensor重設每個型別
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        # n維的x(資料)

        if len(state.shape) == 1:
            # 處理1維的x(資料)
            # 把資料透過torch.unsqueeze轉成1維
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            done = (done,)  # 把done轉為一個資料(tuple)

        # 1. 透過現在的狀態去預測Q值
        pred = self.model(state)

        # 2. Q_new = r + y * max(next_predicted Q value)    ->  only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        target = pred.clone()  # 複製前面的預測
        for idx in range(len(done)):
            Q_new = reward[idx]  # 預設新的Q值為當前編號的reward

            # 如果done為false(遊戲還沒結束，可以預測下一個行動)
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))  # 藉由公式計算新的Q值

            target[idx][torch.argmax(action).item()] = Q_new  # 將運測中行動可能性最高的設為新的Q值

        # 優化
        self.optimizer.zero_grad()  # 初始化計算梯度為0
        loss = self.criterion(target, pred)  # (Q_new, Q)計算損失率
        loss.backward()  # 反向傳播計算梯度

        self.optimizer.step()  # 更新所有參數
