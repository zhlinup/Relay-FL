clear;
clc;

K = 100;
N = 1;

f_c = 915 * 10^6;  % carrier bandwidth
G_A = 4.11;  % antenna gain
PL = 3;  % path loss value
light = 3 * 10^8;  % speed of light

trial = 50;

d_AP = [0, 0];

channel_U = zeros(K, trial);
channel_R = zeros(N, trial);
channel_UR = zeros(K, N, trial);

for i = 1 : trial
    
    dx_U = unifrnd(80, 120, K, 1);
    dy_U = unifrnd(-60, 60, K, 1);

    dx_R = 50;
    dy_R = 0;

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
    g_rayl_R = (randn(N, 1) + 1j * randn(N, 1)) / sqrt(2); 
    g_rayl_UR = (randn(K, N) + 1j * randn(K, N)) / sqrt(2);
    
    channel_U(:, i) = g_rayl_U .* sqrt(PL_U);
    channel_R(:, i) = g_rayl_R .* sqrt(PL_R);
    channel_UR(:, :, i) = g_rayl_UR .* sqrt(PL_UR);
end

filename=['DATA/channel_model_trial_' num2str(trial) '_K_' num2str(K)  '_N_' num2str(N) '_PL_' num2str(PL) '_single.mat'];
save(filename)