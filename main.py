from typing import Optional, Any, Callable

import numpy as np
import argparse
import math
import time
import torch
import copy
import learning_flow
import train_script
from Nets import CNNMnist, MLP
import scipy.io as sio


def initial():
    # network parameters
    setup = argparse.ArgumentParser()
    setup.add_argument('--K', type=int, default=20, help='total # of devices')
    setup.add_argument('--N', type=int, default=1, help='# of relays')
    setup.add_argument('--PL', type=float, default=3.0, help='path loss exponent')

    # simulation parameters
    setup.add_argument('--trial', type=int, default=50, help='# of Monte Carlo Trials')
    setup.add_argument('--SNR', type=float, default=100, help='-noise variance in dB')
    # setup.add_argument('--P_0', type=float, default=0.1, help='user transmit power budget 0.1W')
    setup.add_argument('--P_r', type=float, default=0.1, help='relay transmit power budget 0.1W')
    setup.add_argument('--verbose', type=int, default=1, help=r'whether output or not')
    setup.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # learning parameters
    setup.add_argument('--gpu', type=int, default=0, help=r'Use which gpu')
    setup.add_argument('--local_ep', type=int, default=1, help="the number of local epochs, E")
    setup.add_argument('--local_bs', type=int, default=0, help="0 for no effect, local bath size, B")
    setup.add_argument('--lr', type=float, default=0.05, help="learning rate, lambda")
    setup.add_argument('--low_lr', type=float, default=1e-5, help="learning rate lower bound, bar_lambda")
    setup.add_argument('--gamma', type=float, default=0.9, help="learning rate decrease ratio, gamma")
    setup.add_argument('--step', type=int, default=50, help="learning rate decrease step, bar_T")
    setup.add_argument('--momentum', type=float, default=0.99,
                       help="SGD momentum, used only for multiple local updates")
    setup.add_argument('--epochs', type=int, default=500, help="rounds of training, T")
    setup.add_argument('--iid', type=int, default=1, help="1 for iid, 0 for non-iid")
    setup.add_argument('--noniid_level', type=int, default=2, help="number of classes at each device for non-iid")
    setup.add_argument('--V_idx', type=int, default=4, help="Variable index")
    args = setup.parse_args()
    return args


if __name__ == '__main__':
    setup = initial()
    np.random.seed(setup.seed)
    torch.manual_seed(setup.seed)

    setup.init_lr = copy.deepcopy(setup.lr)

    print(setup)

    # data = sio.loadmat('matlab/DATA/trial_50_K_20_N_1_PL_3_0519.mat')
    data = sio.loadmat('matlab/DATA/trial_50_K_20_N_1_PL_3_Pr.mat')
    Pr_set = [0.01, 0.1, 0.3, 0.5, 1]
    V_idx = setup.V_idx
    setup.P_r = Pr_set[V_idx]

    store_filename = 'store/trial_{}_K_{}_N_{}_B_{}_E_{}_lr_{}_SNR_{}_PL_{}_Pr_{}.npz'.format(setup.trial, setup.K,
                                                                                              setup.N, setup.local_bs,
                                                                                              setup.local_ep, setup.lr,
                                                                                              setup.SNR, setup.PL,
                                                                                              setup.P_r)
    print(store_filename)

    setup.sigma = np.power(10, -setup.SNR / 10)

    channel_U = data['channel_U']
    channel_R = data['channel_R']
    channel_UR = data['channel_UR']

    Proposed_a_k1 = data['Proposed_a_k1']
    Proposed_a_k2 = data['Proposed_a_k2']
    Proposed_b_n = data['Proposed_b_n']
    Proposed_c_1 = data['Proposed_c_1']
    Proposed_c_2 = data['Proposed_c_2']

    Single_a_k1 = data['Single_a_k1']
    Single_c_1 = data['Single_c_1']

    Xu_a_k1 = data['Xu_a_k1']
    Xu_b_n = data['Xu_b_n']
    Xu_eta = data['Xu_eta']

    MSE_1 = np.zeros([setup.trial, 2 * setup.epochs])
    MSE_2 = np.zeros([setup.trial, setup.epochs])
    MSE_3 = np.zeros([setup.trial, 2 * setup.epochs])
    MSE_4 = np.zeros([setup.trial, setup.epochs])
    MSE_5 = np.zeros([setup.trial, setup.epochs])

    MSE2_1 = np.zeros([setup.trial, 2 * setup.epochs])
    MSE2_2 = np.zeros([setup.trial, setup.epochs])
    MSE2_3 = np.zeros([setup.trial, 2 * setup.epochs])
    MSE2_4 = np.zeros([setup.trial, setup.epochs])
    MSE2_5 = np.zeros([setup.trial, setup.epochs])

    result_store = []

    result_set = []
    result_CNN_set = []
    result_CNN_MB_set = []

    print(torch.__version__)

    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    setup.device = torch.device(
        'cuda:{}'.format(setup.gpu) if torch.cuda.is_available() and setup.gpu != -1 else 'cpu')
    print(setup.device)

    for i in range(setup.trial):
        print('This is the {0}-th trial'.format(i))

        setup.h_k = channel_U[: setup.K, i]
        setup.f_n = channel_R[: setup.N, i]
        setup.g_kn = channel_UR[: setup.K, : setup.N, i]

        p_a_k1 = Proposed_a_k1[V_idx, : setup.K, i]
        p_a_k2 = Proposed_a_k2[V_idx, : setup.K, i]
        p_b_n = Proposed_b_n[V_idx, : setup.N, i]
        p_c_1 = Proposed_c_1[V_idx, i]
        p_c_2 = Proposed_c_2[V_idx, i]

        s_a_k1 = Single_a_k1[V_idx, : setup.K, i]
        s_c_1 = Single_c_1[V_idx, i]

        x_a_k1 = Xu_a_k1[V_idx, : setup.K, i]
        x_b_n = Xu_b_n[V_idx, : setup.N, i]
        x_eta = Xu_eta[V_idx, i]

        Error_free = 1
        Proposed = 1
        Single_slot = 1
        Xu_scheme = 1

        result = {}

        if setup.iid:
            train_images, train_labels, test_images, test_labels, size = train_script.load_fmnist_iid(setup.K)
        else:
            train_images, train_labels, test_images, test_labels, size = train_script.load_fmnist_noniid(setup.K,
                                                                                                         setup.non_iid_level)
        net_glob = CNNMnist(num_classes=10, num_channels=1, batch_norm=True).to(setup.device)
        # net_glob = MLP(784, 64, 10).to(setup.device)

        setup.size = size
        setup.rho = np.ones(setup.K, dtype=float) * (setup.size / np.sum(setup.size))
        if setup.verbose:
            print(net_glob)
        w_glob = net_glob.state_dict()
        w_0 = copy.deepcopy(w_glob)
        d = 0
        for item in w_glob.keys():
            d = d + int(np.prod(w_glob[item].shape))
        print('Total Number of Parameters={}'.format(d))

        net_glob.load_state_dict(w_glob)
        idxs_users = np.asarray(range(setup.N))

        if Error_free:
            print('Error_Free Channel is running')
            loss_train1, accuracy_test1, loss_test1, mse_1, mse2_1 = learning_flow.learning_iter(setup, d, net_glob,
                                                                                                 w_glob, train_images,
                                                                                                 train_labels,
                                                                                                 test_images,
                                                                                                 test_labels, 1, None,
                                                                                                 None, None,
                                                                                                 None, None)
            result['loss_train1'] = np.asarray(loss_train1)
            result['accuracy_test1'] = np.asarray(accuracy_test1)
            result['loss_test1'] = np.asarray(loss_test1)
            print('result {}'.format(result['accuracy_test1'][len(result['accuracy_test1']) - 1]))
            MSE_1[i, :] = mse_1
            MSE2_1[i, :] = mse2_1

        if Proposed:
            print('Proposed Scheme is running')

            w_glob = copy.deepcopy(w_0)
            net_glob.load_state_dict(w_glob)

            loss_train2, accuracy_test2, loss_test2, mse_2, mse2_2 = learning_flow.learning_iter(setup, d, net_glob,
                                                                                                 w_glob, train_images,
                                                                                                 train_labels,
                                                                                                 test_images,
                                                                                                 test_labels, 2, p_a_k1,
                                                                                                 p_a_k2, p_b_n,
                                                                                                 p_c_1, p_c_2)

            result['loss_train2'] = np.asarray(loss_train2)
            result['accuracy_test2'] = np.asarray(accuracy_test2)
            result['loss_test2'] = np.asarray(loss_test2)
            print('result {}'.format(result['accuracy_test2'][len(result['accuracy_test2']) - 1]))
            MSE_2[i, :] = mse_2
            MSE2_2[i, :] = mse2_2

        if Single_slot:
            print('Conventional Scheme is running')

            w_glob = copy.deepcopy(w_0)
            net_glob.load_state_dict(w_glob)

            loss_train3, accuracy_test3, loss_test3, mse_3, mse2_3 = learning_flow.learning_iter(setup, d, net_glob,
                                                                                                 w_glob, train_images,
                                                                                                 train_labels,
                                                                                                 test_images,
                                                                                                 test_labels, 3,
                                                                                                 s_a_k1, None,
                                                                                                 None, s_c_1, None)
            result['loss_train3'] = np.asarray(loss_train3)
            result['accuracy_test3'] = np.asarray(accuracy_test3)
            result['loss_test3'] = np.asarray(loss_test3)
            print('result {}'.format(result['accuracy_test3'][len(result['accuracy_test3']) - 1]))
            MSE_3[i, :] = mse_3
            MSE2_3[i, :] = mse2_3

        if Xu_scheme:
            print('Existing Scheme is running')

            w_glob = copy.deepcopy(w_0)
            net_glob.load_state_dict(w_glob)

            loss_train5, accuracy_test5, loss_test5, mse_5, mse2_5 = learning_flow.learning_iter(setup, d, net_glob,
                                                                                                 w_glob, train_images,
                                                                                                 train_labels,
                                                                                                 test_images,
                                                                                                 test_labels, 5,
                                                                                                 x_a_k1,
                                                                                                 None, x_b_n, None,
                                                                                                 x_eta)
            result['loss_train5'] = np.asarray(loss_train5)
            result['accuracy_test5'] = np.asarray(accuracy_test5)
            result['loss_test5'] = np.asarray(loss_test5)
            print('result {}'.format(result['accuracy_test5'][len(result['accuracy_test5']) - 1]))
            MSE_5[i, :] = mse_5
            MSE2_5[i, :] = mse2_5

        result_store.append(result)
        np.savez(store_filename, vars(setup), result_store, MSE_1, MSE_2, MSE_3, MSE_4, MSE_5, MSE2_1, MSE2_2, MSE2_3,
                 MSE2_4, MSE2_5)

    np.savez(store_filename, vars(setup), result_store, MSE_1, MSE_2, MSE_3, MSE_4, MSE_5, MSE2_1, MSE2_2, MSE2_3,
             MSE2_4, MSE2_5)
