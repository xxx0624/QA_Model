# python-LDA

Ref: `https://github.com/a55509432/python-LDA`

the model applies `Sampling` function by Python language.
---

* If you found the result is 0 by Python2.7, it may be a bug to be fixed.

---
### Dada format
#### Train data format
    `train.dat` after segment word, the format like following：（one line one document）
>1. 康小姐 寮步镇 莞樟路 石井 附近 嘉湖山庄 小区 连续 半夜 停电 已有 居民 咨询 供电公司 小区 电路 正常 咨询 小区 管理处 工作人员 线路 借口 推托<br>
>2. 许小姐 来电反映 寮步镇 莞樟路 汽车东站 附近 嘉湖山庄 小区 最近 一周 都 从 凌晨 3点 早上 8点 停电 昨晚 凌晨 来电 都 没 通电 已有 居民 致电 供电公司 答复 说 该 小区 电路 正常 小区 故意 停电 <br>
>3. 虎门 百佳商场 楼下 乘坐 出租车 虎门 电子城 车牌 粤SLE857 司机 要求 不 打表 需要 20元 要求 打表 司机 拒载<br>
>4. 东城中心 乘坐 粤SM643M  东城 主山高田坊芳桂园 平时 行驶 路线 是 东城 中路 今天 司机 行驶 路线 是 东城大道 东纵大道 温南路 此 车 到了 温南路口车费 是  16元 认为 司机 绕路<br>

#### Output data format
>        `model_parameter.dat` the params of the model
>        `wordidmap.dat` the relation between word and id 
>        `model_twords.dat` topN words of every classification
>        `model_tassgin.dat` the result of words which belong to someone classification
>        `model_theta.dat` the possibility of documents which belong to all classifications 
>        `model_phi.dat` the possibility of words which belong to all classifications 

---
### How to run

1. `prepare_lda_data.py` 

    修改其中的train文件位置
    - modify the location of train data
    - update the params of lda and modify the output data of path in `setting.conf`
2.  `lda.py`

    cd data/lda.py
    python lda.py
    
3. `retreat_lda_data.py`

    - modify the path of dim, model and train data files
    - `prepare_lda_data.py` may be skip some wrong data.
    - default value is 0.01 in `retreat_lda_data.py`
    
---

