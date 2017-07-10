__author__ = 'xing'
import os, sys
import codecs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def draw_x_y(x, y, color):
    plt.figure(1)
    plt.plot(x, y, color)
    plt.show()


if __name__ == '__main__':

    with codecs.open(sys.argv[1], 'r', encoding='utf-8') as fr:
        #fw1 = codecs.open(str(sys.argv[1]) + '.little-epoch', 'w', encoding='utf-8')
        #fw2 = codecs.open(str(sys.argv[1]) + '.big-epoch', 'w', encoding='utf-8')

        little_epoch = 0
        little_epoch_list = []
        cost_list = []
        acc_list = []

        big_epoch = 0
        big_epoch_list = []
        top1_list = []
        mrr_list = []
        map_list = []

        for line in fr:
            line = line.strip()
            if 'epoch:' in line and 'cost:' in line and 'acc:' in line:
                little_epoch += 1
                epoch = line.split('epoch:')[1].split(' ')[0]
                cost = line.split('cost:')[1].split(',')[0]
                acc = line.split('acc:')[1]
                #fw1.write(epoch + ',' + cost + ',' + acc + '\n')
                #for draw
                little_epoch_list.append(little_epoch)
                cost_list.append(cost)
                acc_list.append(acc)
            elif 'top-1 = ' in line:
                big_epoch += 1
                top1 = line.split('top-1 = ')[1].split('/')[0].strip()
                mrr = line.split('mrr = ')[1].split('/')[0].strip()
                map = line.split('map = ')[1].strip()
                #fw2.write(top1 + ',' + mrr + ',' + map + '\n')
                #for draw
                big_epoch_list.append(big_epoch)
                mrr_list.append(mrr)
                map_list.append(map)
        #fw1.close()
        #fw2.close()

        #start draw
        #draw_x_y(little_epoch_list, cost_list, 'r')
        #draw_x_y(little_epoch_list, acc_list, 'b')
        #draw_x_y(big_epoch_list, top1_list, 'y')
        #draw_x_y(big_epoch_list, map_list, 'k')
        #draw_x_y(big_epoch_list, mrr_list, 'g')

        plt.subplot(221)
        big_epoch_num = 1000
        map_plot = plt.plot(big_epoch_list[:big_epoch_num], map_list[:big_epoch_num], 'r')
        mrr_plot = plt.plot(big_epoch_list[:big_epoch_num], mrr_list[:big_epoch_num], 'b')
        plt.xlabel('epochs')
        plt.ylabel('value')
        map_patch = mpatches.Patch(color='r', label = 'map')
        mrr_patch = mpatches.Patch(color='b', label = 'mrr')
        plt.legend(handles = [map_patch, mrr_patch])

        plt.subplot(222)
        little_epoch_num = 10000
        cost_plot = plt.plot(little_epoch_list[:little_epoch_num], cost_list[:little_epoch_num], 'r')
        acc_plot = plt.plot(little_epoch_list[:little_epoch_num], acc_list[:little_epoch_num], 'b')
        plt.xlabel('epochs')
        plt.ylabel('value')
        cost_patch = mpatches.Patch(color='r', label = 'cost')
        acc_patch = mpatches.Patch(color='b', label = 'acc')
        plt.legend(handles = [cost_patch, acc_patch])

        plt.show()

