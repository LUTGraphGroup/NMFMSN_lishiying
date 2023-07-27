from model import MatrixFactorization,Re_Matix
# 记录文件，时间等
import os
import time

# sk库 绘制ROC曲线，以及计算其他的指标
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 交叉验证
import random
import math
#导入超参
from hyperparams import *

start_time = time.time()




def convert_to_lists(labels, scores):
    label_list = []
    score_list = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i, j] != 2:
                label_list.append(labels[i, j])
                score_list.append(scores[i, j])
    return label_list, score_list


if __name__ == '__main__':
    circrna_disease_matrix = np.load('associationMatrix.npy')
    circ_sim_matrix = np.load('SC.npy')
    dis_sim_matrix = np.load('SD.npy')

    # 划分训练集
    index_tuple = (np.where(circrna_disease_matrix == 1))
    one_list = list(zip(index_tuple[0], index_tuple[1]))
    random.shuffle(one_list)  # 打乱
    split = math.ceil(len(one_list) / kf)

    # 定义一些列表用于保存评估值
    acc = []
    recall = []
    precision = []
    f1 = []
    aupr = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)


    # 5-fold start
    fold = 1
    for i in range(0, len(one_list), split): 
        print('第{}次交叉验证开始'.format(fold))
        test_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()

        # 抹除已知关系
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0

        # 把训练集的标签改为2
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix

        # 重构邻接矩阵
        re_m = Re_Matix()
        re_circrna_disease_matrix = re_m.process_matrix(new_circrna_disease_matrix, circ_sim_matrix, dis_sim_matrix, k)

        # NMF
        mf = MatrixFactorization()
        C, D = mf.nonnegative_matrix_factorization(re_circrna_disease_matrix, circ_sim_matrix, dis_sim_matrix, r
                                                   , alpha, beta,,gamma, lr, max_iter, tol,diff_threshold)

        prediction_matrix = np.dot(C, D.T)
        # 求fpr，tpr以绘制ROC曲线
        y_true, y_scores = convert_to_lists(roc_circrna_disease_matrix, prediction_matrix)
        print('交叉验证的样本数量：', len(y_true), len(y_scores))
        fpr, tpr, thersholds = roc_curve(y_true, y_scores, pos_label=1)

        # interp:插值 把结果添加到tprs列表中
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        print('AUC：{}'.format(roc_auc))
        aucs.append(roc_auc)

        # 其他指标
        y_pred = [1 if score >= 0.5 else 0 for score in y_scores]
        acc.append(accuracy_score(y_true, y_pred))
        recall.append(recall_score(y_true, y_pred))
        precision.append(precision_score(y_true, y_pred))
        f1.append(f1_score(y_true, y_pred))

        # pr曲线
        pre, rec, prthresholds = precision_recall_curve(y_true, y_scores)
        aupr.append(auc(rec, pre))


        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.4f)' % (fold, roc_auc))

        fold += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    # 计算平均值
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    avg_acc = sum(acc) / len(acc)
    avg_recall = sum(recall) / len(recall)
    avg_precision = sum(precision) / len(precision)
    avg_f1 = sum(f1) / len(f1)
    avg_aupr = sum(aupr) / len(aupr)


    fpr_tpr = np.vstack((mean_fpr, mean_tpr))

    # 打印指标
    print("Auc 平均值为：{:.4f}".format(mean_auc))
    print("Accuracy 平均值为：{:.4f}".format(avg_acc))
    print("Recall 平均值为：{:.4f}".format(avg_recall))
    print("Precision 平均值为：{:.4f}".format(avg_precision))
    print("F1 平均值为：{:.4f}".format(avg_f1))
    print("AUPR 平均值为：{:.4f}".format(avg_aupr))

    # 画图
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='red', label=r'Mean ROC (area=%0.4f)' % mean_auc, lw=1.5, alpha=0.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    # plt.show()

    # 运行时间
    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间为S：", run_time)
