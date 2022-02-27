# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.io as sio

if __name__ == '__main__':
    d = 21921
    trial = 50
    trial2 = 28
    K = 20
    N = 1
    SNR = 100
    B = 0
    E = 1
    lr = 0.05
    PL = 3.0
    P_r = 0.1
    iid = 1
    noniid_level = 2
    Pr_set = [0.01, 0.1, 0.3, 0.5, 1]

    test_accuracy = np.zeros([5, len(Pr_set)])
    training_loss = np.zeros([5, len(Pr_set)])
    nmse = np.zeros([5, len(Pr_set)])

    for i in range(len(Pr_set)):
        P_r = Pr_set[i]
        filename = 'store/trial_{}_K_{}_N_{}_B_{}_E_{}_lr_{}_SNR_{}_PL_{}_Pr_{}.npz'.format(trial, K, N, B, E, lr, SNR,
                                                                                            PL, P_r)

        a = np.load(filename, allow_pickle=True)
        res = a['arr_1']
        nmse1 = a['arr_3']
        nmse2 = a['arr_4']
        nmse4 = a['arr_6']

        nmse[1, i] = 10 * np.log10(np.mean(nmse1[~np.isnan(nmse1)]))
        nmse[2, i] = 10 * np.log10(np.mean(nmse2[~np.isnan(nmse2)]))
        nmse[4, i] = 10 * np.log10(np.mean(nmse4[~np.isnan(nmse4)]))

        res_CNN = {}
        for iter in range(trial2):
            if iter == 0:
                res_CNN = copy.deepcopy(res[0])
            else:
                for item in res_CNN.keys():
                    res_CNN[item] += copy.deepcopy(res[iter][item])

        for item in res_CNN.keys():
            res_CNN[item] = copy.deepcopy(res_CNN[item] / trial2)

        test_accuracy[0, i] = res_CNN['accuracy_test1'][1000]
        test_accuracy[1, i] = res_CNN['accuracy_test2'][500]
        test_accuracy[2, i] = res_CNN['accuracy_test3'][1000]
        test_accuracy[4, i] = res_CNN['accuracy_test5'][500]

    print(test_accuracy)
    print(training_loss)

    matfile = 'matlab/training_result/cmp_Pr_trial_{}_K_{}_N_{}_B_{}_E_{}.mat'.format(trial2, K, N, B, E)
    sio.savemat(matfile, mdict={'test_accuracy': test_accuracy})

    matfile2 = 'matlab/training_result/cmp_Pr_trial_{}_K_{}_N_{}_B_{}_E_{}_NMSE.mat'.format(trial2, K, N, B, E)
    sio.savemat(matfile2, mdict={'nmse': nmse})

    plt.plot(Pr_set, test_accuracy[0], 'r-')
    plt.plot(Pr_set, test_accuracy[1], 'b-')
    plt.plot(Pr_set, test_accuracy[2], 'g-')
    plt.plot(Pr_set, test_accuracy[4], 'y-')
    plt.legend(labels=['Error-Free', 'Proposed', 'Conventional', 'Existing Scheme'], loc='lower center',
               fontsize='x-large')
    plt.xlim([0.01, 1])
    plt.xticks(Pr_set)
    # plt.ylim([0.2, 0.9])
    # plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    plt.xlabel('Maximum relay transmit power $P_r$')
    plt.ylabel('Test Accuracy')
    plt.grid()

    plt.figure()

    plt.plot(Pr_set, nmse[1], 'b-')
    plt.plot(Pr_set, nmse[2], 'g-')
    plt.plot(Pr_set, nmse[4], 'y-')
    plt.legend(labels=['Proposed', 'Conventional', 'Existing Scheme'], loc='lower center',
               fontsize='x-large')
    plt.xlim([0.01, 1])
    plt.xticks(Pr_set)
    # plt.ylim([0.2, 0.9])
    # plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    plt.xlabel('Maximum relay transmit power $P_r$')
    plt.ylabel('Average MSE')

    plt.show()
