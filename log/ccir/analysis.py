import os, sys

def get_arrs(line):
    if '=' in line:
        return line.strip().split('=')[1].strip()
    else:
        return 'None'


def get_avg_dcg_list(line):
    avg_dcg3 = line.strip().split('AVG-DCG3 = ')[1].split(' / ')[0]
    avg_dcg5 = line.strip().split('AVG-DCG5 = ')[1].split(' / ')[0]
    avg_dcg = line.strip().split(' = ')[-1]
    return avg_dcg3, avg_dcg5, avg_dcg


def analysis(filepath):
    res = {}
    with open(filepath, 'r') as fr:
        for line in fr:
            value = get_arrs(line)
            if 'batch_size' in line:
                res['batch_size'] = value
            elif 'filter_sizes' in line:
                res['filter_sizes'] = value
            elif 'num_filters' in line:
                res['num_filters'] = value
            elif 'embedding_size' in line:
                res['embedding_size'] = value
            elif 'learning_rate' in line:
                res['learning_rate'] = value
            elif 'n_epochs' in line:
                res['n_epochs'] = value
            elif 'validation_freq' in line:
                res['validation_freq'] = value
            elif 'keep_prob_value' in line:
                res['keep_prob_value'] = value
            elif 'margin_size' in line:
                res['margin_size'] = value
            elif 'words_num_dim' in line:
                res['words_num_dim'] = value
            elif 'keep_prob_value' in line:
                res['keep_prob_value'] = value
            elif 'train_1_file' in line:
                res['train_1_file'] = value
            elif 'train_0_file' in line:
                res['train_0_file'] = value
            elif 'test_file' in line:
                res['test_file'] = value
            elif 'vector_file' in line:
                res['vector_file'] = value
            #find max
            elif 'AVG-DCG3' in line and 'AVG-DCG5' in line and 'AVG-DCG' in line:
                try:
                    dcg3, dcg5, dcg = get_avg_dcg_list(line)
                    if 'AVG-DCG3' in res:
                        res['AVG-DCG3'] = max(res['AVG-DCG3'], float(dcg3))
                        res['AVG-DCG5'] = max(res['AVG-DCG5'], float(dcg5))
                        res['AVG-DCG'] = max(res['AVG-DCG'], float(dcg))
                    else:
                        res['AVG-DCG3'] = float(dcg3)
                        res['AVG-DCG5'] = float(dcg5)
                        res['AVG-DCG'] = float(dcg)
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
