import numpy as np
import copy
import torch
import train_script
import AirComp


def FedAvg_grad(w_glob, grad, device):
    ind = 0
    w_return = copy.deepcopy(w_glob)

    for item in w_return.keys():
        a = np.array(w_return[item].size())
        if len(a):
            b = np.prod(a)
            w_return[item] = copy.deepcopy(w_return[item]) + torch.from_numpy(
                np.reshape(grad[ind: ind + b], a)).float().to(device)
            ind = ind + b
    return w_return


def learning_iter(setup, d, net_glob, w_glob, train_images, train_labels, test_images, test_labels,
                  trans_mode, a_k1, a_k2, b_n, c_1, c_2):
    loss_train = []
    mse_train = []
    mse2_train = []
    accuracy_test = []
    loss_test_set = []

    net_glob.eval()
    acc_test, loss_test = train_script.test_model(net_glob, setup, test_images, test_labels)
    accuracy_test.append(acc_test)

    net_glob.train()

    if trans_mode == 1 or trans_mode == 3:
        epochs = setup.epochs * 2
    else:
        epochs = setup.epochs

    setup.lr = setup.init_lr
    for iter in range(epochs):
        if iter > 1 and iter % setup.step == 0:
            setup.lr = max(setup.lr * setup.gamma, setup.low_lr)

        gradient_store_per_iter = np.zeros([setup.K, d])

        loss_locals = []
        ind = 0
        for idx in range(setup.K):
            if setup.local_bs == 0:
                size = int(setup.size[idx])
            else:
                size = min(int(setup.size[idx]), setup.local_bs)

            w, loss, gradient = train_script.local_update(setup, d, copy.deepcopy(net_glob).to(setup.device),
                                                          train_images, train_labels, idx, size)

            loss_locals.append(copy.deepcopy(loss))

            copy_g = copy.deepcopy(w)
            copy_g[np.isnan(copy_g)] = 0

            gradient_store_per_iter[ind, :] = copy_g
            ind = ind + 1

        if trans_mode == 1:
            grad = np.average(copy.deepcopy(gradient_store_per_iter), axis=0, weights=setup.rho)
            mse = 0
            mse2 = 0

        elif trans_mode == 2:
            grad, _, mse, mse2 = AirComp.AM(setup, d, copy.deepcopy(gradient_store_per_iter),
                                            a_k1, a_k2, b_n, c_1, c_2)

        elif trans_mode == 3:
            grad, _, mse, mse2 = AirComp.Single(setup, d, copy.deepcopy(gradient_store_per_iter), a_k1, c_1)

        elif trans_mode == 5:
            grad, _, mse, mse2 = AirComp.Xu(setup, d, copy.deepcopy(gradient_store_per_iter), a_k1, b_n, c_2)

        # if setup.verbose:
        #     print(10 * np.log10(mse))
        #     print(10 * np.log10(mse2))
        #     print(np.mean(np.abs(gradient_store_per_iter) ** 2))
        #     print(np.mean(np.abs(grad) ** 2))

        w_glob = copy.deepcopy(FedAvg_grad(w_glob, grad, setup.device))
        net_glob.load_state_dict(w_glob)
        # loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if setup.verbose:
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        loss_train.append(loss_avg)
        mse_train.append(mse)
        mse2_train.append(mse2)

        acc_test, loss_test = train_script.test_model(net_glob, setup, test_images, test_labels)
        accuracy_test.append(acc_test)
        loss_test_set.append(loss_test)
        net_glob.train()

    return loss_train, accuracy_test, loss_test_set, mse_train, mse2_train
