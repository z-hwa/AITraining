import matplotlib.pyplot as slt
import os


def slot(scores, mean_scores, file_name):
    slt.title(file_name)  # 標題
    slt.xlabel('Number of Games')  # x軸
    slt.ylabel('Score')  # y軸
    slt.plot(scores)  # 根據分數繪製圖
    slt.plot(mean_scores)  # 根據平均分數繪製圖
    slt.ylim(ymin=0)  # 設置y最小值
    slt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # 文字顯示
    slt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))


# 儲存訓練圖以及分數資料
def slotSave(file_name, score, mean_score, total_score, record, n_games):
    # 開檔
    path = './data/' + file_name + '.txt'
    f = open(path, 'w')

    # 處理資料格式 轉成str list
    str_score = [str(i) for i in score]
    str_mean_score = [str(i) for i in mean_score]

    # 製作成用,隔開的string
    s = ','.join(str_score)
    ms = ','.join(str_mean_score)

    # 所有string資料 放進list
    data_list = [s, ms, str(total_score), str(record), str(n_games)]
    data = '\n'.join(data_list)  # 透過換行分隔每種資料

    # 寫入
    f.write(data)
    f.close()

    # 訓練圖
    fig = slt.gcf()
    fig.savefig('./figure' + '\\' + file_name + '.png')
    slt.close('all')
