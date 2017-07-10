python-LDA

`https://github.com/a55509432/python-LDA`

============
### lda模型的python实现，算法采用sampling抽样
---

* 项目基于python2.7.10如果发现计算概率为0，可能是python的兼容性问题，暂时没时间修复（发现python3.0以上版本会出现此问题）

---
### 训练和输出文本格式说明
#### 模型训练文件
    `train.dat` 用其他软件or算法分词后，再剔除停用词的最后结果文件，显示格式如下：（一行表示一篇文档）
>1. 康小姐 寮步镇 莞樟路 石井 附近 嘉湖山庄 小区 连续 半夜 停电 已有 居民 咨询 供电公司 小区 电路 正常 咨询 小区 管理处 工作人员 线路 借口 推托<br>
>2. 许小姐 来电反映 寮步镇 莞樟路 汽车东站 附近 嘉湖山庄 小区 最近 一周 都 从 凌晨 3点 早上 8点 停电 昨晚 凌晨 来电 都 没 通电 已有 居民 致电 供电公司 答复 说 该 小区 电路 正常 小区 故意 停电 <br>
>3. 虎门 百佳商场 楼下 乘坐 出租车 虎门 电子城 车牌 粤SLE857 司机 要求 不 打表 需要 20元 要求 打表 司机 拒载<br>
>4. 东城中心 乘坐 粤SM643M  东城 主山高田坊芳桂园 平时 行驶 路线 是 东城 中路 今天 司机 行驶 路线 是 东城大道 东纵大道 温南路 此 车 到了 温南路口车费 是  16元 认为 司机 绕路<br>

#### 模型输出文件
>        `model_parameter.dat` 保存模型训练时选择的参数 
>        `wordidmap.dat` 保存词与id的对应关系，主要用作topN时查询 
>        `model_twords.dat` 输出每个类高频词topN个 
>        `model_tassgin.dat` 输出文章中每个词分派的结果，文本格式为词id:类id 
>        `model_theta.dat` 输出文章与类的分布概率，文本一行表示一篇文章，概率1   概率2 ...表示文章属于类的概率 
>        `model_phi.dat` 输出词与类的分布概率，是一个K*M的矩阵，其中K为设置分类的个数，M为所有文章的词的总数，

---
### 使用说明

1. `prepare_lda_data.py` 

    修改其中的train文件位置
    修改`setting.conf`中的生成的文件的路径以及lda参数
2.  `lda.py`

    cd到data/lda.py
    直接python lda.py
    
3. `retreat_lda_data.py`

    修改其中的dim, model文件位置, train文件位置<br>
    `prepare_lda_data.py` 在处理train数据时，有可能遇到某些不合法语句跳过，故生成的lda数据出现缺失现象<br>
    `retreat_lda_data.py` 补充缺失的数据(default 0.01)
    
---

