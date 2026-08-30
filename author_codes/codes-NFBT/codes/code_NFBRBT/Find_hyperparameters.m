clc;
clear all
close all

N = 256; % the number of the antennas at the BS
K = 1;% the number of users
M = 1;% number of subcarriers
L = 1; % number of paths per user

A = 2;

fc = 100e9; % carrier frequency
Rmin = 4;
Rmax = 4;
sector = pi/6;

fs = 100e6; % bandwidth
Q = 64;  % number of pilot blocks
tmax = 20e-9; % maximum of the path delay
f = zeros(1,M);
for m = 1:M
    f(m)=fc+fs/(M)*(m-1-(M-1)/2);
end
c = 3e8;
lambda_c = c/fc;
d = lambda_c / 2;
eps = 1e-3;


sample = 20;

SNR_dB = 10:2:30;
SNR_linear = 10.^(SNR_dB/10.);


% generate the far-field codebook 
disp('generate the far-field codebook')

s = 4;
D = s*N; 
Codebook_far = zeros(D,N);
col = -1 + 2/D : 2/D : 1 ;
theta_fn = asin(col);
for i = 1:D
    Codebook_far(i,:) = array_respones(theta_fn(i),N,d,lambda_c);
end
S = size(Codebook_far,1);

x = zeros(S,1);
for xx = 1:S
    x(xx) = xx;
end
disp('the far-field codebook has been generated')
     


    X = Codebook_far;
    YY = zeros(S,1000);
    for i = 1:10000
       H = farfield_channel(N,K,L,lambda_c,d);
       Gain_vector = conj(Codebook_far)*H;
       YY(:,i) = abs(Gain_vector); 
    end
    y = mean(YY,2);


    initialParams = [1, 1]; % 初始：[sigma_f, l]
    lb = [1e-5, 1e-5]; % 参数的下界
    ub = [200, 200]; % 参数的上界
    options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'sqp');
    
    [optParams, ~] = fmincon(@(params) gprObjectiveFunction(params, X, y), ...
                             initialParams, [], [], [], [], lb, ub, [], options);

    % 输出最优超参数
    fprintf('Optimized sigma_f: %f\n', optParams(1));
    fprintf('Optimized l: %f\n', optParams(2));

function negLogLikelihood = gprObjectiveFunction(params, X, y)
    sigma_f = params(1);
    l = params(2);
    % 计算指数核函数矩阵
    N = size(X, 1);
    K = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K(i, j) = sigma_f^2 * exp(-norm(X(i,:)-X(j,:))^2 / (2*l^2));
        end
    end
    % 添加噪声项
    sigma_n = 0.1; % 假定噪声方差，根据实际情况调整
    K = K + sigma_n^2 * eye(N);
    % 计算负对数似然
    % 使用 Cholesky 分解来避免直接计算逆和行列式
    L = chol(K, 'lower');
    alpha = L'\(L\y);
    negLogLikelihood = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*N*log(2*pi);
   
end






