# QA-Model

2017 `NLPCC` DBQA 比赛：3rd(`MRR=0.685011`)

2017 `CCIR` 比赛：9th(`NDCG@3=0.7006 NDCG@5=0.7398`)

# Run
    Step1: corpus/nlpcc16-dbqa-data/train中readme

    Step2: corpus/nlpcc16-dbqa-data/test中readme

    Step3: cnn/theano/中readme
    
<br>
Tips：目前在处理train和test数据时候用的分词是cnn.theano.util.word_segment.segment_word_filter_pos,
如果需要更改可在corpus/nlpcc16-dbqa-data/train和corpus/nlpcc16-dbqa-data/test中的脚本自行替换函数
(可替换为cnn.theano.util.word_segment.segment_word)