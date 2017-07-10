# train

## 1在本目录下
    修改你的QAModel位置：sys.path.append('/home/zhengxing/QAModel')

## 2在本目录下
    filepath = '.' //源文件位置
    
    filename = 'train' //源文件名
    
    filename1 = filename + '1_data_version1' //正样本数据文件名
    
    filename0 = filename + '0_data_version1' //正样本数据文件名


## Tips:

    1.  raw data is `train.1.json`
    2.  `expand_raw_data.py` is used to expand the raw data with crawling raw
    text from the url, then generate the new raw data called 'train.1.json.new'
    3.  `prepare_train_test_data.py` is used to generate the 2/3 train data 
    and 1/3 test data
    4.  `prepare_expanded_URL_train_test_data.py` is used to generate the 2/3 
    train data (expanded with the title of the url) and 1/3 test data 
    (expanded with the title of the url)
    5.  `prepare_train_data.py` is not used.

