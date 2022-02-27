import numpy as np


def AM(setup, d, signal, a_k1, a_k2, b_n, c_1, c_2):
    rho = setup.rho

    g_mean = np.mean(signal, axis=1)
    global_g_mean = rho.T @ g_mean

    g_var = np.var(signal, axis=1)
    global_g_var = rho.T @ g_var

    var_mean_sqrt = global_g_var ** 0.5

    noise_1 = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2) * np.sqrt(setup.sigma)
    noise_2 = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2) * np.sqrt(setup.sigma)
    noise_N = (np.random.randn(setup.N, d) + 1j * np.random.randn(setup.N, d)) / np.sqrt(2) * np.sqrt(setup.sigma)

    x_signal_1 = np.tile(a_k1, (d, 1)).T * (signal - np.tile(global_g_mean, (d, 1)).T) / var_mean_sqrt
    x_signal_2 = np.tile(a_k2, (d, 1)).T * (signal - np.tile(global_g_mean, (d, 1)).T) / var_mean_sqrt

    r_n = setup.g_kn.T @ x_signal_1 + noise_N

    y_1 = setup.h_k.T @ x_signal_1 + noise_1
    y_2 = setup.h_k.T @ x_signal_2 + setup.f_n.T @ (np.tile(b_n, (d, 1)).T * r_n) + noise_2

    w = np.real((y_1 * c_1 + y_2 * c_2) * var_mean_sqrt + global_g_mean)
    true_w = rho.T @ signal
    avg_mse = np.linalg.norm((true_w - w)) ** 2 / np.linalg.norm(true_w) ** 2
    mse2 = np.linalg.norm((true_w - w)) ** 2 / d
    return w, true_w, avg_mse, mse2


def Single(setup, d, signal, a_k1, c_1):
    rho = setup.rho

    g_mean = np.mean(signal, axis=1)
    global_g_mean = rho.T @ g_mean

    g_var = np.var(signal, axis=1)
    global_g_var = rho.T @ g_var

    var_mean_sqrt = global_g_var ** 0.5

    noise_1 = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2) * np.sqrt(setup.sigma)

    x_signal = np.tile(a_k1, (d, 1)).T * (signal - np.tile(global_g_mean, (d, 1)).T) / var_mean_sqrt

    y = setup.h_k.T @ x_signal + noise_1

    w = np.real(y * c_1 * var_mean_sqrt + global_g_mean)
    true_w = rho.T @ signal
    avg_mse = np.linalg.norm((true_w - w)) ** 2 / np.linalg.norm(true_w) ** 2
    mse2 = np.linalg.norm((true_w - w)) ** 2 / d
    return w, true_w, avg_mse, mse2


def Xu(setup, d, signal, a_k1, b_n, c_2):
    rho = setup.rho

    g_mean = np.mean(signal, axis=1)
    global_g_mean = rho.T @ g_mean

    g_var = np.var(signal, axis=1)
    global_g_var = rho.T @ g_var

    var_mean_sqrt = global_g_var ** 0.5

    # noise_1 = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2) * np.sqrt(setup.sigma)
    noise_2 = (np.random.randn(d) + 1j * np.random.randn(d)) / np.sqrt(2) * np.sqrt(setup.sigma)
    noise_N = (np.random.randn(setup.N, d) + 1j * np.random.randn(setup.N, d)) / np.sqrt(2) * np.sqrt(setup.sigma)

    x_signal_1 = np.tile(a_k1, (d, 1)).T * (signal - np.tile(global_g_mean, (d, 1)).T) / var_mean_sqrt

    r_n = setup.g_kn.T @ x_signal_1 + noise_N

    y = setup.f_n.T @ (np.tile(b_n, (d, 1)).T * r_n) + noise_2

    w = np.real(y / c_2 * var_mean_sqrt + global_g_mean)
    true_w = rho.T @ signal
    avg_mse = np.linalg.norm((true_w - w)) ** 2 / np.linalg.norm(true_w) ** 2
    mse2 = np.linalg.norm((true_w - w)) ** 2 / d
    return w, true_w, avg_mse, mse2
