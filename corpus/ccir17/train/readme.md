# Tips of preparing data scripts
1.  raw data is `train.1.json`
2.  `expand_raw_data.py` is used to expand the raw data with crawling raw
text from the url, then generate the new raw data called `train.1.json.new`
3.  `prepare_train_test_data.py` is used to generate the 2/3 train data and 1/3 test data
4.  `prepare_expanded_URL_train_test_data.py` is used to generate the 2/3 train data(expanded with the title of the url) and 1/3 test data(expanded with the title of the url)
5.  `prepare_train_data.py` is not used.

# Tips of variables in script
- filepath = '.' // current folder
- filename = 'train' // file name
- filename1 = filename + '1_data_version1' // file name of positive data
- filename0 = filename + '0_data_version1' // file name of negative data