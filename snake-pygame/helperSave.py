import matplotlib.pyplot as slt

def slot(scores, mean_scores):
    slt.title('Training...')  # 標題
    slt.xlabel('Number of Games')  # x軸
    slt.ylabel('Score')  # y軸
    slt.plot(scores)  # 根據分數繪製圖
    slt.plot(mean_scores)  # 根據平均分數繪製圖
    slt.ylim(ymin=0)  # 設置y最小值
    slt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # 文字顯示
    slt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

# 儲存訓練圖
def slotSave(file_name):
    fig = slt.gcf()
    fig.savefig('./figure' + '\\' + file_name + '.png')
    slt.close('all')
