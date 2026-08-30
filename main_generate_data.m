%% main_generate_data.m
% 生成 hybrid-field channel data for LLM-empowered near-field communications
% 按照论文原文 + 作者NFNOMA代码的信道模型重写
%
% 论文: "LLM-Empowered Near-Field Communications for Low-Altitude Economy"
%
% 输出: Data_user.mat
%   h_near_slant    - (total_samples, N) 复数信道矩阵 (未归一化，含路径损耗)
%   index_far_near   - (total_samples, 1) 远/近场分类 (0/1, 基于ENFR准则)

clear; clc; close all;

%% 系统参数 (论文表I + 第五节-A)
N = 256;                    % ULA天线数
K = 10;                     % 每样本用户数
Ns = 10000;                 % 总样本数 (8000训练 + 1000验证 + 1000测试)

fc = 30e9;                  % 载频 30 GHz
c = 3e8;                    % 光速
lambda = c / fc;            % 波长 0.01 m
d = lambda / 2;             % 天线间距 0.5 cm

hB = 15;                    % 基站天线高度 (m), 论文第二节
theta_tilt = 5 * pi / 180;  % 下倾角 (rad), 论文第二节

% 用户坐标范围 (论文第五节-A)
x_min = 0;  x_max = 200;    % 水平距离 (m)
h_min = 0;  h_max = 30;     % 高度 (m)

% ENFR阈值 (论文图4, Delta=0.1)
Delta = 0.1;

% 信道多径参数 (从作者NFNOMA代码推断, 论文未明确给出)
L = 5;                      % NLoS路径数
kappa = 8;                  % Rician因子
sigma_aod = 5 * pi / 180;   % AOD扩展 (rad)

fprintf('系统参数:\n');
fprintf('  N=%d, K=%d, Ns=%d\n', N, K, Ns);
fprintf('  fc=%.0f GHz, lambda=%.4f m, d=%.4f m\n', fc/1e9, lambda, d);
fprintf('  hB=%d m, tilt=%.1f deg\n', hB, theta_tilt*180/pi);
fprintf('  L=%d, kappa=%d, sigma_aod=%.1f deg\n', L, kappa, sigma_aod*180/pi);

%% 预分配
total = Ns * K;
h_near_slant = complex(zeros(total, N));
index_far_near = zeros(total, 1);

nn = -(N-1)/2 : (N-1)/2;    % 天线索引

%% 主循环
fprintf('\n开始生成 %d 样本 x %d 用户 = %d 信道...\n', Ns, K, total);

row = 1;
for n_sample = 1:Ns
    if mod(n_sample, 500) == 0
        fprintf('  %d / %d\n', n_sample, Ns);
    end

    for k = 1:K
        %% 1. 随机用户坐标 (论文第二节, 第五节-A)
        x_k = x_min + (x_max - x_min) * rand;
        h_k = h_min + (h_max - h_min) * rand;

        % 斜距 (BS天线中心到用户)
        dz = h_k - hB;
        r0 = sqrt(x_k^2 + dz^2);

        % 垂直角 (相对于水平面)
        theta_user = atan2(dz, x_k);

        % 相对于下倾角的角度
        theta = theta_user - theta_tilt;

        %% 2. ENFR分类 (论文公式7-9)
        % 近场导向矢量 b(theta, r): 论文公式(2)
        %   b_n = (1/sqrt(N)) * exp(-j*2*pi*(r_n - r)/lambda)
        %   r_n = sqrt(r^2 + x_n^2 - 2*r*x_n*sin(theta))
        % 远场导向矢量 a(theta): 论文公式(3)的共轭形式
        %   a_n = (1/sqrt(N)) * exp(+j*2*pi*d*n*sin(theta)/lambda)
        %   注: 论文定义 a(theta) = (1/sqrt(N)) exp(-j*2*pi*d*n*sin(theta)/lambda)
        %       但 |b^H a| = |b^H conj(a)|，正/负号不影响ENFR判定，
        %       此处取共轭形式以便与 MATLAB 转置运算配合。
        %
        % ENFR准则 (论文公式7-9):
        %   近场条件: 1 - |b^H * a| >= Delta
        %   即 |b^H * a| <= 1 - Delta 时判为近场

        r_elem = sqrt(r0^2 - 2*r0*nn*d*sin(theta) + (nn*d).^2);
        b_nf = exp(-1j*2*pi/lambda*(r_elem - r0)) / sqrt(N);  % 近场导向矢量 b(theta,r)
        b_ff = exp(1j*2*pi/lambda*(nn*d*sin(theta))) / sqrt(N);  % 远场导向矢量 conj(a(theta))

        % 计算 |b^H * a|: 使用 MATLAB 共轭转置 b_ff' 对 b_ff 取共轭
        % [修正] 旧代码 b_nf * b_ff.' 使用了无共轭的转置 ('.'), 导致 |b^H a| ≈ 0
        %        正确应使用共轭转置 b_ff' (单引号), 等价于 b_nf * conj(b_ff)
        bf_gain = abs(b_nf * b_ff');  % |b^H a|
        bf_loss = 1 - bf_gain;

        % 近场: 波束成形增益损失 >= Delta
        index_far_near(row) = double(bf_loss >= Delta);

        %% 3. 信道生成 (作者generate_user_in_circle_multipath.m, 无路径损耗)
        % LoS路径: sqrt(kappa/(kappa+1)) * b(theta, r)
        h_user = sqrt(kappa / (kappa + 1)) * b_nf;

        % NLoS路径: sqrt(1/(kappa+1)) * alpha_l * b(theta_l, r_l) / sqrt(L)
        % 作者代码: alpha = 1, NLoS gain = ssf * sqrt(1/L) * sqrt(1/(1+kappa))
        theta_nlos = theta + sigma_aod * randn(1, L);
        r_nlos = r0 .* (1 + 0.02 * randn(1, L));
        r_nlos = max(r_nlos, 0.1);

        % 衰落系数 (作者: ssf = (randn+1j*randn)/sqrt(2) * sqrt(1/L))
        alpha = (randn(1, L) + 1j * randn(1, L)) / sqrt(2) / sqrt(L);

        for l = 1:L
            r_l_elem = sqrt(r_nlos(l)^2 - 2*r_nlos(l)*nn*d*sin(theta_nlos(l)) + (nn*d).^2);
            b_l = exp(-1j*2*pi/lambda*(r_l_elem - r_nlos(l))) / sqrt(N);
            h_user = h_user + sqrt(1/(kappa+1)) * alpha(l) * b_l;
        end

        % 直接存储，不归一化 (论文信道模型不含路径损耗，也不做归一化)
        h_near_slant(row, :) = h_user;

        row = row + 1;
    end
end

%% 保存
save_path = fullfile(fileparts(mfilename('fullpath')), 'Data_user.mat');
save(save_path, 'h_near_slant', 'index_far_near');

fprintf('\n保存完成: %s\n', save_path);
fprintf('  h_near_slant: [%d x %d] 复数 (未归一化)\n', size(h_near_slant));
fprintf('  index_far_near: [%d x %d]\n', size(index_far_near));
fprintf('  近场比例: %.1f%%\n', 100*mean(index_far_near));
fprintf('  远场比例: %.1f%%\n', 100*(1-mean(index_far_near)));
fprintf('  信道范数: min=%.4f, max=%.4f, mean=%.4f\n', ...
    min(abs(h_near_slant(:))), max(abs(h_near_slant(:))), mean(abs(h_near_slant(:))));
