模型：
default：
    epsilon = 80 (if rand(0~200) < (80 - n): 隨機行動)
    撞、超時懲罰 = -10
    吃到獎勵 = 10

d:DEBUG

a:最原始的訓練參數 11-256-3

b:多了每節身體(50)的方位 61-256-3
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
    -4
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

c:多了每節身體(6)的方位 17-256-3, speed = 80
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        200+原地繞圈

e:多了每節身體(4)的方位 15-256-3, speed = 80
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        成長速度太慢

f:改為紀錄身體(10)的xy座標, speed = 80
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        200+一直轉圈
    -BUG
        因為其他數據都在0~1之間，那麼如果其他數據遠遠大於0~1
        最終取出來的行動方案，會被其他數據大大影響
        修正此問題
    -2
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        狀態良好
        更新:以gpu運算
fa:改為紀錄身體(10) + 食物的xy座標, speed = 80
    -1
        epsilon = 200 - ngames / (bodylengths - 2) (每一次長大 都應該有更多隨機的機會)
        改成以上界epsilon和AI操作率 來決定隨機性
        =
        更新:以gpu運算