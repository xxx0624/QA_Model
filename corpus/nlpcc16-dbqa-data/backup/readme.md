# backup train
Tips：针对原数据train新增的目录，这里的数据是经过人工删选的新数据，格式同原数据train相同

## 1在本目录下

    修改你的QAModel位置：sys.path.append('/home/zhengxing/QAModel')

## 2在本目录下
    filepath = '.' //源文件位置
    
    filename = 'train' //源文件名
    
    filename1 = filename + '1_data_version1' //正样本数据文件名
    
    filename0 = filename + '0_data_version1' //正样本数据文件名


## Tips:

    1. 原始数据：`train`
    2. `expand_tfidf.py` 针对`train`中的每一条数据中的answer进行计算它的每个词语的idf,
    并将idf值前top_k的补充在answer末尾。结果产生3个文件：
        train.word-idf是每个词语的idf值 // word:score
        train.expand-top_k-idf:新的train文件，格式同train
        train.expand-top_k-idf-bk:对应train文件中的每一行，每一行的每个单词的idf值 
    
    3. `prepare_train_data.py` 产生符合格式的新train数据