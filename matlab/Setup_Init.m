function setup = Setup_Init(K, N, h_k, f_n, g_kn, P_r)

%--------------------------------------------------------------------------
%System model Parameters
setup.K = K;
setup.N = N;

setup.D = ones(setup.K, 1) / setup.K;
setup.rho = ones(setup.K, 1) / setup.K;

setup.P_K = ones(setup.K, 1) * P_r;
setup.P_N = ones(setup.N, 1) * 0.1;

setup.P_0 = 0.1;
setup.P_r = P_r;

setup.SNR = 100;
setup.sigma_0 = power(10, -setup.SNR / 10);
setup.noise_N = ones(setup.N, 1) * setup.sigma_0;

setup.sigma = power(10, -setup.SNR / 10);

setup.J_max = 100;
setup.threshold = 1e-4;

setup.h_k = h_k;
setup.f_n = f_n;
setup.g_kn = g_kn;

