clear
clc
tic

load('DATA/channel_model_trial_50_K_100_N_1_PL_3_single'); % single-relay
% load('DATA/channel_model_trial_50_K_100_N_1_PL_3_loc'); % single relay location
% load('DATA/channel_model_trial_50_K_50_N_4_PL_3'); % single-cell network
% load('DATA/channel_model_trial_50_K_50_N_64_PL_3'); % single-cell network with 64 relays

K = 20;
N = 1;
trial = 50;
PL = 3;

V_set = [0.01 0.1 0.3 0.5 1.0];

V_length = length(V_set);

Proposed_nmse = zeros(V_length, trial);
Single_nmse = zeros(V_length, trial);
Xu_nmse = zeros(V_length, trial);

Proposed_mse = zeros(V_length, trial);
Single_mse = zeros(V_length, trial);
Xu_mse = zeros(V_length, trial);

Proposed_mmse = zeros(V_length, trial);
Single_mmse = zeros(V_length, trial);
Xu_mmse = zeros(V_length, trial);

Proposed_a_k1 = zeros(V_length, K, trial);
Proposed_a_k2 = zeros(V_length, K, trial);
Proposed_b_n = zeros(V_length, N, trial);
Proposed_c_1 = zeros(V_length, trial);
Proposed_c_2 = zeros(V_length, trial);

Proposed_ite = zeros(V_length, trial);

Single_a_k1 = zeros(V_length, K, trial);
Single_c_1 = zeros(V_length, trial);

Xu_a_k1 = zeros(V_length, K, trial);
Xu_b_n = zeros(V_length, N, trial);
Xu_eta = zeros(V_length, trial);

d = 100000;
signal = normrnd(0, 1, [K, d]);
grad = mean(signal, 1);

filename=['DATA/trial_' num2str(trial) '_K_' num2str(K)  '_N_' num2str(N) '_PL_' num2str(PL) '_Pr_source.mat'];

for V_idx = 1 : V_length
    
    P_r = V_set(V_idx);
   
    parfor iter = 1 : trial % parallel computing
%     for iter = 1 : trial
        fprintf('%d-th trial\n', iter);

        h_k = channel_U(1 : K, iter);
        f_n = channel_R(1 : N, iter);
        g_kn = channel_UR(1 : K, 1 : N, iter);

        setup = Setup_Init(K, N, h_k, f_n, g_kn, P_r);
        
        tic;
        t_start = cputime;
    
        [w1, true_w1, Single_nmse(V_idx, iter), Single_mse(V_idx, iter), Single_mmse(V_idx, iter), Single_a_k1(V_idx, :, iter), Single_c_1(V_idx, iter)] = Single(setup, d, signal);
        [w3, true_w3, Xu_nmse(V_idx, iter), Xu_mse(V_idx, iter), Xu_mmse(V_idx, iter), Xu_a_k1(V_idx, :, iter), Xu_b_n(V_idx, :, iter), Xu_eta(V_idx, iter)] = Xu(setup, d, signal);
        [w4, true_w4, Proposed_ite(V_idx, iter), Proposed_nmse(V_idx, iter), Proposed_mse(V_idx, iter), Proposed_mmse(V_idx, iter), Proposed_a_k1(V_idx, :, iter), Proposed_a_k2(V_idx, :, iter), Proposed_b_n(V_idx, :, iter), Proposed_c_1(V_idx, iter), Proposed_c_2(V_idx, iter)] = AM(setup, d, signal);
        
        t_end = cputime;
        toc_end = toc;
    end
    save(filename);
end