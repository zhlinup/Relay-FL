function [w, true_w, ave_mse, mse, MMSE, a_k1, b_n, eta] = Xu(setup, d, signal)

g_mean = mean(signal, 2);
global_g_mean = setup.rho.' * g_mean;

g_var = var(signal, 0, 2);
global_g_var = setup.rho.' * g_var;

var_mean_sqrt = sqrt(global_g_var);

h_k = setup.h_k;
f_n = setup.f_n;
g_kn = setup.g_kn;

rho = setup.rho;
P_0 = setup.P_0;
P_r = setup.P_r;
sigma = setup.sigma;

K = setup.K;
N = setup.N;

rx_scaling = sqrt(P_0) * abs(h_k) ./ rho;

eta = min(rx_scaling);

a_k1 = eta * rho ./ h_k;

b_n = zeros(N, 1);
for n = 1 : N
    b_n(n) = sqrt(P_r / (transpose(abs(g_kn(:, n)).^2) * abs(a_k1).^2 + sigma));
end

obj = norm(g_kn * (f_n .* b_n) .* a_k1 / eta - rho)^2 ...
    + (1 + transpose(abs(f_n).^2) * abs(b_n).^2) * sigma / abs(eta)^2;

obj_pre = 1e6;
threshold = setup.threshold;

scaling_factor = 1e2;
scaling_factor2 = 1e-2;

obj_vec = [obj];

while (obj_pre - obj) / obj_pre > threshold
    
    obj_pre = obj;
    
    a_angle = - angle(g_kn * (f_n .* b_n));
    
    hat_Pr = P_r ./ (abs(b_n).^2) - sigma;
    
    cvx_begin quiet
        variable a_vec1(K);
        minimize(square_pos(norm(scaling_factor * abs(g_kn * (f_n .* b_n)) .* a_vec1 / eta - scaling_factor * rho)));
        subject to
            for k = 1 : K
                a_vec1(k) <= sqrt(P_0);
            end
            for n = 1 : N
                transpose(scaling_factor * abs(g_kn(:, n)).^2) * power(a_vec1, 2) <= scaling_factor * hat_Pr(n);
            end
    cvx_end
    cvx_status;
    
    if strcmp(cvx_status, 'Failed') ~= 1
        a_k1 = a_vec1 .* exp(1j * a_angle);
    else
        cvx_begin quiet
            variable a_vec1(K);
            minimize(square_pos(norm(abs(g_kn * (f_n .* b_n)) .* a_vec1 / eta - rho)));
            subject to
                for k = 1 : K
                    a_vec1(k) <= sqrt(P_0);
                end
                for n = 1 : N
                    transpose(abs(g_kn(:, n)).^2) * power(a_vec1, 2) <= hat_Pr(n);
                end
        cvx_end
        cvx_status
        if strcmp(cvx_status, 'Failed') ~= 1
            a_k1 = a_vec1 .* exp(1j * a_angle);
        else
            cvx_begin quiet
                variable a_vec1(K);
                minimize(square_pos(norm(scaling_factor2 * abs(g_kn * (f_n .* b_n)) .* a_vec1 / eta - scaling_factor2 * rho)));
                subject to
                    for k = 1 : K
                        a_vec1(k) <= sqrt(P_0);
                    end
                    for n = 1 : N
                        transpose(scaling_factor2 * abs(g_kn(:, n)).^2) * power(a_vec1, 2) <= scaling_factor2 * hat_Pr(n);
                    end
            cvx_end
            cvx_status
            if strcmp(cvx_status, 'Failed') ~= 1
                a_k1 = a_vec1 .* exp(1j * a_angle);
            end
        end
    end
    
    bar_Pr = zeros(N, 1);
    a1_g = zeros(K, N);
    for k = 1 : K
        a1_g(k, :) = a_k1(k) * transpose(g_kn(k, :));
    end
    tmp2 = 0;  
    
    
    tmp1 = (rho .* a_k1).' * g_kn;
    for k = 1 : K
        tmp2 = tmp2 + abs(a_k1(k))^2 * g_kn(k, :)' * g_kn(k, :);
    end
    
    b_vec = eta * ((scaling_factor^4 * (tmp2 + sigma * eye(N)) * diag(f_n)) \ (scaling_factor^4 * tmp1)');

    for n = 1 : N
        tmp3 = 0;
        
        for k = 1 : K
            tmp3 = tmp3 + abs(g_kn(k, n))^2 * abs(a_k1(k))^2;
        end
        hat_bn = b_vec(n);
        bar_Pr(n) = P_r / (tmp3 + sigma);
        
        if abs(hat_bn) >= sqrt(bar_Pr(n))
            b_n(n) = sqrt(bar_Pr(n)) * hat_bn / norm(hat_bn);
        else
            b_n(n) = hat_bn;
        end
    end
    
    eta = (b_n' * diag(f_n)' * (tmp2 + sigma * eye(N)) * diag(f_n) * b_n + sigma) / (transpose(rho .* a_k1) * g_kn * diag(f_n) * b_n);
    
    obj = norm(g_kn * (f_n .* b_n) .* a_k1 / eta - rho)^2 ...
    + (1 + transpose(abs(f_n).^2) * abs(b_n).^2) * sigma / abs(eta)^2;

    obj_vec = [obj_vec, obj];
end

noise_2 = (randn(1, d) + 1j * randn(1, d)) / sqrt(2) * sqrt(sigma);
noise_N = (randn(N, d) + 1j * randn(N, d)) / sqrt(2) * sqrt(sigma);

x_signal_1 = repmat(a_k1, 1, d) .* ((signal - global_g_mean) / var_mean_sqrt);

r_n= transpose(g_kn) * x_signal_1 + noise_N;

y = setup.f_n.' * (b_n .* r_n) + noise_2;

w = real(y / eta * var_mean_sqrt + global_g_mean);
true_w = rho.' * signal;
ave_mse = norm(true_w - w)^2 / norm(true_w)^2;

mse = norm(true_w - w)^2 / d;

rho_hat = g_kn * (f_n .* b_n) .* a_k1 / eta;
MMSE = norm(g_kn * (f_n .* b_n) .* a_k1 / eta - rho)^2 ...
    + (1 + transpose(abs(f_n).^2) * abs(b_n).^2) * sigma / abs(eta)^2;    