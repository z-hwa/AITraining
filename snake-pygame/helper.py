import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # 開啟plot的互動模式


def plot(scores, mean_scores):
    display.clear_output(wait=True)  # 清空輸出
    display.display(plt.gcf())  # 獲取當前數字
    plt.clf()  # 清除當前數字
    plt.title('Training...')  # 標題
    plt.xlable('Number of Games')  # x軸
    plt.ylable('Score')  # y軸
    plt.plot(scores)  # 根據分數繪製圖
    plt.plot(mean_scores)  # 根據平均分數繪製圖
    plt.ylim(ymin=0)  # 設置y最小值
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # 文字顯示
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
