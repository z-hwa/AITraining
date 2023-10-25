# AI訓練統整專案

## 作者資料
王冠章  
2023年 成功大學數學系 大一  

## 目錄說明
  
存放在snake-pygame/README.md當中  
  
## 使用注意事項
  
agent訓練以及模型實測相關的操作時，需要確保agent、agentPlaying中，模型的輸入層數量正確  
並且在上述兩個腳本的Agent class的get_state函數，所蒐集的資料必須修改成相對應的資料  
已訓練好的模型，所使用的訓練資料，都記錄在record_model_train.md中  
  
## 基於anaconda的虛擬環境

1. 如果沒有anaconda，參考 https://zhuanlan.zhihu.com/p/459607806 下載  
2. 記得init激活base環境  

## 環境：
1. python 3.7  
2. pygame  
3. power shell 7.3  
4. pytorch with cpu, gpu  
5. matplotlib ipython  

## 參考資料：
### snakegame ai
author：Patrick Loeber  
project name：Teach AI To Play Snake - Reinforcement Learning Tutorial With PyTorch And Pygame  
ref：https://youtu.be/5Vy5Dxu7vDs  

## DEBUG:
1. 套件發生衝突: https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a  