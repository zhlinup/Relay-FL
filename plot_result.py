import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy
import scipy.io as sio

if __name__ == '__main__':
    trial = 50
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
    loc = 50

    kappa = 0.4

    filename = 'store/trial_{}_K_{}_N_{}_B_{}_E_{}_lr_{}_SNR_{}_PL_{}_Pr_{}.npz'.format(trial, K, N, B, E, lr, SNR, PL,
                                                                                        P_r)

    print(filename)

    nmse = np.zeros(5)

    a = np.load(filename, allow_pickle=1)
    # result_CNN_set2 = a['arr_1']

    result_CNN_set = a['arr_1']
    result_MSE_set = a['arr_2']
    result_NMSE_set = a['arr_2']

    nmse1 = a['arr_3']
    nmse2 = a['arr_4']
    nmse4 = a['arr_6']

    trial2 = 30
    # for i in range(trial):
    for i in range(trial2):
        if i == 0:
            res_CNN = copy.deepcopy(result_CNN_set[0])
        else:
            for item in res_CNN.keys():
                res_CNN[item] += copy.deepcopy(result_CNN_set[i][item])

    for item in res_CNN.keys():
        # res_CNN[item] = copy.deepcopy(res_CNN[item] / trial)
        res_CNN[item] = copy.deepcopy(res_CNN[item] / trial2)

    test_accuracy1 = res_CNN['accuracy_test1']
    test_accuracy2 = res_CNN['accuracy_test2']
    test_accuracy3 = res_CNN['accuracy_test3']
    test_accuracy5 = res_CNN['accuracy_test5']

    nmse[1] = 10 * np.log10(np.mean(nmse1[~np.isnan(nmse1)]))
    nmse[2] = 10 * np.log10(np.mean(nmse2[~np.isnan(nmse2)]))
    nmse[4] = 10 * np.log10(np.mean(nmse4[~np.isnan(nmse4)]))

    matfile = 'matlab/training_result/cmp_time_trial_{}_K_{}_N_{}_B_{}_E_{}.mat'.format(trial2, K, N, B, E)
    sio.savemat(matfile, mdict={'test_accuracy1': test_accuracy1[0: 1001], 'test_accuracy2': test_accuracy2[0: 501],
                                'test_accuracy3': test_accuracy3[0: 1001], 'test_accuracy5': test_accuracy5[0: 501]})
    matfile2 = 'matlab/training_result/cmp_time_trial_{}_K_{}_N_{}_B_{}_E_{}_NMSE.mat'.format(trial, K, N, B, E)
    sio.savemat(matfile2, mdict={'nmse': nmse})

    plt.plot(np.arange(0, len(test_accuracy1)), test_accuracy1, 'k--', label=r'Error-Free Channel')
    plt.plot(np.arange(0, 2 * len(test_accuracy2), 2), test_accuracy2, '-o', markersize=6, markevery=100,
             label=r'Proposed Scheme')
    plt.plot(np.arange(0, len(test_accuracy3)), test_accuracy3, '-*', markersize=8, markevery=100,
             label=r'Conventional')
    plt.plot(np.arange(0, 2 * len(test_accuracy5), 2), test_accuracy5, '->', markersize=6, markevery=100,
             label=r'Existing Scheme')

    plt.legend()
    plt.xlim([0, 1000])
    plt.ylim([0, 0.9])
    plt.xlabel('Transmission Time Slot')
    plt.ylabel('Test Accuracy')
    plt.grid()
    # plt.show()

    # plt.figure()
    # # plt.plot(np.arange(0, len(MSE_1)), MSE_1, 'r--')
    # plt.plot(np.arange(0, 2 * len(MSE_2), 2), MSE_2, 'b-')
    # plt.plot(np.arange(0, len(MSE_3)), MSE_3, 'g-')
    # plt.plot(np.arange(0, 2 * len(MSE_4), 2), MSE_4, 'm--')
    # plt.plot(np.arange(0, 2 * len(MSE_5), 2), MSE_5, 'y--')
    # plt.legend(labels=['Proposed', 'Conventional', 'Double-Slot', 'Existing Scheme'], loc='lower right',
    #            fontsize='x-large')
    # plt.xlabel('Transmission Time Slot')
    # plt.ylabel('NMSE')
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(np.arange(0, 2 * len(MSE2_2), 2), MSE2_2, 'b-')
    # plt.plot(np.arange(0, len(MSE2_3)), MSE2_3, 'g-')
    # plt.plot(np.arange(0, 2 * len(MSE2_4), 2), MSE2_4, 'm--')
    # plt.plot(np.arange(0, 2 * len(MSE2_5), 2), MSE2_5, 'y--')
    # plt.legend(labels=['Proposed', 'Conventional', 'Double-Slot', 'Existing Scheme'], loc='lower right',
    #            fontsize='x-large')
    # plt.xlabel('Transmission Time Slot')
    # plt.ylabel('MSE')
    # plt.grid()
    plt.show()
