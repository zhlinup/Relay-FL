clear;
clc;

K = 50;
N = 64;

f_c = 915 * 10^6;  % carrier bandwidth
G_A = 4.11;  % antenna gain
PL = 3;  % path loss value
light = 3 * 10^8;  % speed of light

trial = 50;

d_AP = [0, 0];

channel_U = zeros(K, trial);
channel_R = zeros(N, trial);
channel_UR = zeros(K, N, trial);

r_min = 0;
r_max = 120;

A = 2 / (r_max * r_max - r_min * r_min);

relay_r = ones(N, 1) * 50;
relay_theta = zeros(N, 1);  
for n = 1 : N
    relay_theta(n) = (n - 1) * 2 * pi / N;
end

ini = 1;
relay_theta2 = [];
for i = 1 : log2(64) + 1
    relay_theta2 = [relay_theta2, relay_theta(1 : 64 / ini : 64)];
    ini = ini * 2;
end

dx_R = relay_r .* cos(relay_theta);
dy_R = relay_r .* sin(relay_theta);

rng(1)
user_theta_set = unifrnd(0, 2 * pi, K, 1, trial);  
user_r_set = sqrt(2 .* unifrnd(0, 1, K, 1, trial) / A + r_min * r_min);

dx_U = user_r_set .* cos(user_theta_set);
dy_U = user_r_set .* sin(user_theta_set);

% rng('shuffle')

for i = 1 : trial
           
    user_theta = user_theta_set(:, :, 1);  
    user_r = user_r_set(:, :, 1);
    
    dx_U = user_r .* cos(user_theta);
    dy_U = user_r .* sin(user_theta);

    dis_U = sqrt((dx_U - d_AP(1)).^2 + (dy_U - d_AP(2)).^2);
    dis_R = sqrt((dx_R - d_AP(1)).^2 + (dy_R - d_AP(2)).^2);
    dis_UR = zeros(K, N);

    for k = 1 : K
        for n = 1 : N
            dis_UR(k, n) = sqrt((dx_U(k) - dx_R(n))^2 + (dy_U(k) - dy_R(n))^2); 
        end
    end

    PL_U = G_A * (light ./ (4 * pi * f_c .* dis_U)).^PL;
    PL_R = G_A * (light ./ (4 * pi * f_c .* dis_R)).^PL;
    PL_UR = G_A * (light ./ (4 * pi * f_c .* dis_UR)).^PL;
    
    g_rayl_U = (randn(K, 1) + 1j * randn(K, 1)) / sqrt(2);    %Rayleigh fading component for K
    g_rayl_R = (randn(64, 1) + 1j * randn(64, 1)) / sqrt(2); 
    g_rayl_UR = (randn(K, 64) + 1j * randn(K, 64)) / sqrt(2);
    
    channel_U(:, i) = g_rayl_U .* sqrt(PL_U);
    channel_R(:, i) = g_rayl_R(1 : 64 / N : 64) .* sqrt(PL_R);
    channel_UR(:, :, i) = g_rayl_UR(:, 1 : 64 / N : 64) .* sqrt(PL_UR);
end

filename=['DATA/channel_model_trial_' num2str(trial) '_K_' num2str(K)  '_N_' num2str(N) '_PL_' num2str(PL) '.mat'];
save(filename)
