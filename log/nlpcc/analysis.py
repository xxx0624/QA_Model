import os, sys

def get_arrs(line):
    if '=' in line:
        return line.strip().split('=')[1].strip()
    else:
        return 'None'


def get_top1(line):
    if 'top-1 = ' in line:
        return line.strip().split('top-1 = ')[1].strip().split('/')[0].strip()
    return '-1'


def get_mrr(line):
    if 'mrr = ' in line:
        return line.strip().split('mrr = ')[1].strip().split('/')[0].strip()
    return '-1'


def get_map(line):
    if 'map = ' in line:
        return line.strip().split('map = ')[1].strip()
    return '-1'


def analysis(filepath):
    res = {}
    with open(filepath, 'r') as fr:
        for line in fr:
            value = get_arrs(line)
            if 'batch_size = ' in line:
                res['batch_size'] = value
            elif 'filter_sizes = ' in line:
                res['filter_sizes'] = value
            elif 'num_filters = ' in line:
                res['num_filters'] = value
            elif 'embedding_size = ' in line:
                res['embedding_size'] = value
            elif 'learning_rate = ' in line:
                res['learning_rate'] = value
            elif 'n_epochs = ' in line:
                res['n_epochs'] = value
            elif 'validation_freq = ' in line:
                res['validation_freq'] = value
            elif 'keep_prob_value = ' in line:
                res['keep_prob_value'] = value
            elif 'margin_size = ' in line:
                res['margin_size'] = value
            elif 'words_num_dim = ' in line:
                res['words_num_dim'] = value
            elif 'keep_prob_value = ' in line:
                res['keep_prob_value'] = value
            elif 'train_1_file = ' in line:
                res['train_1_file'] = value
            elif 'train_0_file = ' in line:
                res['train_0_file'] = value
            elif 'test_file = ' in line:
                res['test_file'] = value
            elif 'vector_file = ' in line:
                res['vector_file'] = value
            elif 'idf_file_path = ' in line:
                res['idf_file_path'] = value
            elif 'lda_train_file_path = ' in line:
                res['lda_train_file_path'] = value
            elif 'lda_test_file_path = ' in line:
                res['lda_test_file_path'] = value
            #find max
            elif 'top-1 = ' in line:
                try:
                    if 'map' not in res:
                        res['map'] = float(get_map(line))
                        res['top-1'] = float(get_top1(line))
                        res['mrr'] = float(get_mrr(line))
                    else:
                        if float(get_map(line)) > float(res['map']):
                            res['map'] = float(get_map(line))
                            res['top-1'] = float(get_top1(line))
                            res['mrr'] = float(get_mrr(line))
                except ValueError:
                    print '[Error]:', filepath
    return res

if __name__ == '__main__':
    if len(sys.argv) == 2:
        print analysis(sys.argv[1])
    else:
        log_file_path = os.getcwd()
        for one_item in os.listdir(log_file_path):
            if 'log' in one_item:
                print one_item
                print analysis(one_item)
                print '\n'
