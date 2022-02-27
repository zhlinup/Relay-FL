function [w, true_w, ite, ave_mse, mse, MMSE, a_k1, a_k2, b_n, c_1, c_2] = AM(setup, d, signal)

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

rx_scaling = sqrt(2) / sqrt(P_0) * rho ./ abs(h_k) / 2;

c_1 = max(rx_scaling);
c_2 = c_1;

a_k1 = rho ./ h_k / c_1 / 2;
a_k2 = a_k1;

b_n = zeros(N, 1);
for n = 1 : N
    b_n(n) = sqrt(P_r / (transpose(abs(g_kn(:, n)).^2) * abs(a_k1).^2 + sigma));
end

ga_m = abs(g_kn).^2;

theta = c_1 * h_k + c_2 * g_kn * (f_n .* b_n);
phi = c_2 * h_k;

rho_hat = theta .* a_k1 + phi .* a_k2;
obj = norm(theta .* a_k1 + phi .* a_k2 - rho)^2 ...
    + (abs(c_1)^2 + abs(c_2)^2 * transpose(abs(f_n).^2) * abs(b_n).^2 ...
    + abs(c_2)^2) * sigma;

J_max = setup.J_max;
threshold = setup.threshold;
obj_vec = [obj];

scaling_factor = 1e2;
scaling_factor2 = 1e-2;

for j = 1 : J_max
    obj_pre = obj;
    
    hat_Pr = P_r ./ abs(b_n).^2 - sigma;

    theta = c_1 * h_k + c_2 * g_kn * (f_n .* b_n);
    phi = c_2 * h_k;

    cvx_begin quiet
        variable a_vec1(K) complex;
        variable a_vec2(K) complex;
        minimize(square_pos(norm(scaling_factor * theta .* a_vec1 + scaling_factor * phi .* a_vec2 - scaling_factor * rho)));
        subject to
            for k = 1 : K
                pow_abs(a_vec1(k), 2) <= P_0 / 2;
                pow_abs(a_vec2(k), 2) <= P_0 / 2;
            end
            for n = 1 : N
                transpose(scaling_factor * abs(g_kn(:, n)).^2) * pow_abs(a_vec1, 2) <= scaling_factor * hat_Pr(n);
            end
    cvx_end
    cvx_status;
    
    if strcmp(cvx_status, 'Failed') ~= 1
        a_k1 = a_vec1;
        a_k2 = a_vec2;
    else
        cvx_begin quiet
            variable a_vec1(K) complex;
            variable a_vec2(K) complex;
            minimize(square_pos(norm(theta .* a_vec1 + phi .* a_vec2 - rho)));
            subject to
                for k = 1 : K
                    pow_abs(a_vec1(k), 2) <= P_0 / 2;
                    pow_abs(a_vec2(k), 2) <= P_0 / 2;
                end
                for n = 1 : N
                    transpose(abs(g_kn(:, n)).^2) * pow_abs(a_vec1, 2) <= hat_Pr(n);
                end
        cvx_end
        cvx_status
        if strcmp(cvx_status, 'Failed') ~= 1
            a_k1 = a_vec1;
            a_k2 = a_vec2;
        else
            cvx_begin quiet
                variable a_vec1(K) complex;
                variable a_vec2(K) complex;
                minimize(square_pos(norm(scaling_factor2 * theta .* a_vec1 + scaling_factor2 * phi .* a_vec2 - scaling_factor2 * rho)));
                subject to
                    for k = 1 : K
                        pow_abs(a_vec1(k), 2) <= P_0 / 2;
                        pow_abs(a_vec2(k), 2) <= P_0 / 2;
                    end
                    for n = 1 : N
                        transpose(scaling_factor2 * abs(g_kn(:, n)).^2) * pow_abs(a_vec1, 2) <= scaling_factor2 * hat_Pr(n);
                    end
            cvx_end
            cvx_status
            if strcmp(cvx_status, 'Failed') ~= 1
                a_k1 = a_vec1;
                a_k2 = a_vec2;
            end
        end
    end

    cons_cm = rho - c_1 * h_k .* a_k1 - c_2 * h_k .* a_k2;
    bar_Pr = zeros(N, 1);
    a1_g = zeros(K, N);
    for k = 1 : K
        a1_g(k, :) = a_k1(k) * transpose(g_kn(k, :));
    end
    tmp1 = 0;
    tmp2 = 0;   
    for k = 1 : K     
        tmp1 = tmp1 + cons_cm(k) * ctranspose(a1_g(k, :));
        tmp2 = tmp2 + abs(a_k1(k))^2 * (g_kn(k, :)' * g_kn(k, :));
    end
    
    b_vec = ((tmp2 + sigma * eye(N)) * diag(f_n)* c_2) \ (tmp1);

    for n = 1 : N
        tmp3 = 0;
        
        for k = 1 : K
            tmp3 = tmp3 + abs(g_kn(k, n))^2 * abs(a_k1(k))^2;
        end
        hat_bn = b_vec(n);
        bar_Pr(n) = P_r / (tmp3 + sigma);
        
        if abs(hat_bn) > sqrt(bar_Pr(n))
            b_n(n) = sqrt(bar_Pr(n)) * hat_bn / norm(hat_bn);
        else
            b_n(n) = hat_bn;
        end
    end

    
    tmp1 = 0;
    tmp2 = 0;
    tmp3 = g_kn * (f_n .* b_n) .* a_k1;

    for k = 1 : K
        tmp1 = tmp1 + (rho(k) - c_2 * (tmp3(k) + h_k(k) * a_k2(k))) * conj(h_k(k) * a_k1(k));
        tmp2 = tmp2 + abs(h_k(k))^2 * abs(a_k1(k))^2;
    end
    
    c_1 = tmp1 / (tmp2 + sigma);
    

    tmp1 = 0;
    tmp2 = 0;
    tmp4 = sigma * transpose(abs(f_n).^2) * abs(b_n).^2;

    for k = 1 : K
        tmp1 = tmp1 + conj(tmp3(k) + h_k(k) * a_k2(k)) * (rho(k) - c_1 * h_k(k) * a_k1(k));
        tmp2 = tmp2 + abs(tmp3(k) + h_k(k) * a_k2(k))^2;
    end
    
    c_2 = tmp1 / (tmp2 + tmp4 + sigma);

    
    theta = c_1 * h_k + c_2 * g_kn * (f_n .* b_n);
    phi = c_2 * h_k;
    
    obj = norm(theta .* a_k1 + phi .* a_k2 - rho)^2 ...
        + (abs(c_1)^2 + abs(c_2)^2 * transpose(abs(f_n).^2) * abs(b_n).^2 ...
        + abs(c_2)^2) * sigma;

    
    if abs(obj - obj_pre) / abs(obj) <= threshold
        break
    end
    
    obj_vec = [obj_vec, obj];
end

ite = j;

noise_1 = (randn(1, d) + 1j * randn(1, d)) / sqrt(2) * sqrt(sigma);
noise_2 = (randn(1, d) + 1j * randn(1, d)) / sqrt(2) * sqrt(sigma);

noise_N = (randn(N, d) + 1j * randn(N, d)) / sqrt(2) * sqrt(sigma);

x_signal_1 = repmat(a_k1, 1, d) .* ((signal - global_g_mean) / var_mean_sqrt);
x_signal_2 = repmat(a_k2, 1, d) .* ((signal - global_g_mean) / var_mean_sqrt);

r_n= transpose(g_kn) * x_signal_1 + noise_N;

y_1 = setup.h_k.' * x_signal_1 + noise_1;
y_2 = setup.h_k.' * x_signal_2 + setup.f_n.' * (b_n .* r_n) + noise_2;

w = real((y_1 * c_1 + y_2 * c_2) * var_mean_sqrt + global_g_mean);
true_w = rho.' * signal;
ave_mse = norm(true_w - w)^2 / norm(true_w)^2;

mse = norm(true_w - w)^2 / d;

theta = c_1 * h_k + c_2 * g_kn * (f_n .* b_n);
phi = c_2 * h_k;

rho_hat = theta .* a_k1 + phi .* a_k2;
MMSE = norm(rho_hat - rho)^2 ...
        + (abs(c_1)^2 + abs(c_2)^2 * transpose(abs(f_n).^2) * abs(b_n).^2 ...
        + abs(c_2)^2) * sigma; 


