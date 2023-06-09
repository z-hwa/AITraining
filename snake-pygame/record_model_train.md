模型訓練日誌：
時間序(由上而下)

Original(最初版本)：
    -layer
        存在危險的行為(前進、右轉、左轉)、相鄰方向是否碰撞(r, d, u, l)、食物方位(左、右、上、下)
        256
        三種行為(前進、右轉、左轉)
    -information
        epsilon = 80 (if rand(0~200) < (80 - n): 隨機行動)
        撞、超時懲罰 = -10
        吃到獎勵 = 10
        speed = 80

d(debug):
    -1
        GPU訓練需要將所有張量、神經網路、損失算法轉成在同一個設備執行
        Ex. model.cuda()
    -2
        先將包含numpy.ndarrays的list轉成np.array再轉成tensor會更快
    -3
	    繼續訓練 資料儲存以及載入的功能測試

u(update):
    -1
        2023.06.08
        增加GPU訓練功能
        並改用GPU運算
    -2
        2023.06.09
        在儲存訓練圖的時候，會同時儲存訓練資料
        因此，繼續訓練時，將能延續上次的訓練圖以及資料，繼續訓練模型

a:最原始的訓練參數 11-256-3
    -layer
        存在危險的行為(前進、右轉、左轉)、相鄰方向是否碰撞(r, d, u, l)、食物方位(左、右、上、下)
        256
        三種行為(前進、右轉、左轉)
    -1
    -2
    -3-u3
        重新訓練的對照組

b:多了每節身體(50)的方位 61-256-3
    -layer
        存在危險的行為(前進、右轉、左轉)
        相鄰方向是否碰撞(r, d, u, l)
        食物方位(左、右、上、下)
        當前身體前一個身體的方位(r,l,u,d)*50: 對應1, 2, 3, 4(頭為0)
    -2
        epsilon = 500 - ngames(6倍輸入層)
        超時懲罰 = 10.2(不要撞自己)
        =
        500+開始不斷繞圈
    -3
        epsilon = 200 * (bodylenths - 2) - ngames(每一次長大 都應該有更多隨機的機會)
        吃到獎勵 = 10.2(鼓勵吃東西)
        =
        500+開始不斷繞圈
    -BUGS
        epsilon大於200以後，會完全只進行隨機的操作，所以前兩個版本的AI根本沒多少進步機會 
        更新(u1)
    -4-u1
        epsilon = 500 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        吃到獎勵 = 10.2(鼓勵吃東西)
        =
        500+還是不斷繞圈
    -5
        upper epsilon = 80
        epsilon = 80 - ngames / (bodylengths - 2)
        (if rand(...) < (...) + 10: 永遠保留隨機行動)
        吃到獎勵 = 10.2(鼓勵吃東西)
        =
        80+不斷繞圈
    -BUGS(d1)
        我的body_dir會在傳遞資料時 一起被改變 因此訓練會失敗
        是因為body_dir出問題了
        更正為透過list.copy來複製

c:多了每節身體(6)的方位 17-256-3
    -layer
        存在危險的行為(前進、右轉、左轉)
        相鄰方向是否碰撞(r, d, u, l)
        食物方位(左、右、上、下)
        當前身體前一個身體的方位(r,l,u,d)*50: 對應1, 2, 3, 4(頭為0)
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        200+原地繞圈

e:多了每節身體(4)的方位 15-256-3
    -layer
        存在危險的行為(前進、右轉、左轉)
        相鄰方向是否碰撞(r, d, u, l)
        食物方位(左、右、上、下)
        當前身體前一個身體的方位(r,l,u,d)*50: 對應1, 2, 3, 4(頭為0)
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        成長速度太慢

f:改為紀錄身體(10)的xy座標
    -layer
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        當前身體xy座標(x, y) * 10:實際座標數字、不存在就是-0.00001
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        200+一直轉圈
    -BUG
        因為其他數據都在0~1之間，那麼如果其他數據遠遠大於0~1
        最終取出來的行動方案，會被其他數據大大影響
        修正此問題
        下方座標範圍(0~1)
    -2
        座標數字/最大長寬 把數值壓縮在0~1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        狀態良好
        更新:以gpu運算(u1)
    -3
        epsilon = 80 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
	    因為前期的繞圈行為太多
	    更新:以GPU運算(u1)

fa:改為紀錄身體(10) + 食物的xy座標
    -layer
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        當前身體xy座標(x, y) * 10: 座標數字/最大長寬 把數值壓縮在0~1、不存在就是-0.00001
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        更新:以gpu運算(u1)

fb:改為紀錄身體*10的xy座標
    -layer -u1
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        當前身體xy座標(x, y) * 10: 座標數字/最大長寬 把數值壓縮在0~1、不存在就是-0.00001
    -1
        epsilon = upperepsilon - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
	    upper epsilon = 80
	    =
	    更新 完成繼續訓練的資料紀錄功能
    -2
        epsilon = upperepsilon - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
	    upper epsilon = 80
	    =
	    更新 完成繼續訓練的資料紀錄功能
        當訓練量超過2000以後，AI實際行動中幾乎不會出現撞到自己前10個身體的狀況

f10:改為紀錄身體*10的xy座標
    -layer -u2
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        當前身體xy座標(x, y) * 10: 座標數字/最大長寬 把數值壓縮在0~1、不存在就是-0.00001
    -1
        upper epsilon = 217 (一個輸入層*7)
        epsilon = upper epsilon - ngames
        punish step -0.01
        =
        最終平均得分收斂在趨於20左右的地方
        儘管AI行為上 較少自撞
        卻會選擇原地繞圈(記錄前10節身體 學會不要撞自己) 或是在遊戲剛開始的時候 加速撞向牆面而死(步數懲罰?)

f30:改為紀錄身體*30的xy座標
    -layer -u2
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        當前身體xy座標(x, y) * 30: 座標數字/最大長寬 把數值壓縮在0~1、不存在就是-0.00001
    -1
        upper epsilon = 497 (一個輸入層*7)
        epsilon = upper epsilon - ngames
        punish step -0.01
        =
        6000回也只有20多的效果
        訓練的成長幅度也趨緩
        推測給予身體節數的配置不佳

g:記錄頭向四面延伸，是否打到身體*4 這個距離是多少*4
    -layer -u2
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        頭向外延伸射線(左、右、上、下)是否打中: 0 或 1
        距離幾格打中(沒打中=0): 格子數量
    -1
        upper epsilon = 133 (一個輸入層*7)
        epsilon = upper epsilon - ngames
        移動步數不應該給懲罰(有的時候必須繞路 才不會困住自己)
        =
        誤判自己跟食物的距離
        可能會出現某個方向 距離更多格才打中 但卻是死路(距離比較少格的才是活路)
        但整體效果上超越了參考的配置 5 左右

h:記錄頭向四面延伸，是否打到身體*4 這個距離是多少*4 記錄頭在xy水平線上 距離食物多遠*2
    -layer -u2
        存在危險的行為(前進、右轉、左轉): 0 或 1
        相鄰方向是否碰撞(r, d, u, l): 0 或 1
        食物方位(左、右、上、下):0 或 1
        頭向外延伸射線(左、右、上、下)是否打中: 0 或 1
        距離幾格打中(沒打中=0): 格子數量
        距離食物位置xy: 格子數量
    -1
        upper epsilon = 147 (一個輸入層*7)
        epsilon = upper epsilon - ngames
        移動步數不應該給懲罰(有的時候必須繞路 才不會困住自己)
        =
        會想要一直降低跟食物的距離，從而選擇把自己困死的路
        整體成效不佳