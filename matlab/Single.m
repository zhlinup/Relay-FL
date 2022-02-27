function [w, true_w, ave_mse, mse, MMSE, tx_scaling, rx_scaling_opt] = Single(setup, d, signal)

g = signal;
g_mean = mean(signal, 2);
global_g_mean = setup.rho.' * g_mean;

g_var = var(signal, 0, 2);
global_g_var = setup.rho.' * g_var;

var_mean_sqrt = sqrt(global_g_var);

rx_scaling = 1 / sqrt(setup.P_0) * setup.rho ./ abs(setup.h_k);

rx_scaling_opt = max(rx_scaling);

tx_scaling = setup.rho ./ setup.h_k / rx_scaling_opt;

noise_1 = (randn(1, d) + 1j * randn(1, d)) / sqrt(2) * sqrt(setup.sigma);

x_signal = repmat(tx_scaling, 1, d) .* ((signal - global_g_mean) / var_mean_sqrt);
y = setup.h_k.' * x_signal + noise_1;

w = real(y * rx_scaling_opt * var_mean_sqrt + global_g_mean);
true_w = setup.rho.' * signal;
ave_mse = norm(true_w - w)^2 / norm(true_w)^2;

mse = norm(true_w - w)^2 / d;

rho_hat = tx_scaling .* setup.h_k * rx_scaling_opt;
MMSE = norm(rho_hat - setup.rho)^2 + setup.sigma * rx_scaling_opt^2;

aa = 1;