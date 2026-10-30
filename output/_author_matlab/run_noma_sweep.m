function run_noma_sweep(ch_file, out_file)
% 官方 NF-NOMA 调用链（main:168-190 同款）在项目固定信道上跑 K 扫描 + P 扫描，
% 与论文 Data.xlsx 的 NF-NOMA 曲线对比。设置不变：QuaCode 码本、NF_group_1→NF_NOMA、
% Imax=30、σ²=0.01；仅评估点为自变量。n_sub=每点样本数（取固定前 n_sub 个）。
% 只读诊断驱动，官方文件零改动。
off = 'C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\codes-NFNOMA\codes-NFNOMA';
cd(off); addpath(off, '-begin');

N = 256; d = 0.005; lam = 0.01; delta = 1.8; rho_min = 4;
Imax = 30; sigma2 = 0.01;

data = load(ch_file);
H_batch = data.H_batch;          % [1000,K,N] complex
n_sub = 200;
H_all = H_batch(1:n_sub, :, :);

[w_near, ~, ~] = QuaCode(N, d, lam, delta, rho_min);
wr = reshape(w_near, N, []);

K_list = 5:10;                   % Fig.6 轴（论文：K 从 5 增至 10）
P_list = 10.^((-10:2:10)/10);    % Fig.9 轴（论文：-10~10 dBW）

SE_k = nan(length(K_list), n_sub); TR_k = nan(length(K_list), n_sub);
SE_p = nan(length(P_list), n_sub); TR_p = nan(length(P_list), n_sub);

t_all = tic;
fprintf('== sweep start: n_sub=%d, Imax=%d ==\n', n_sub, Imax);

for i = 1:length(K_list)
    K_ = K_list(i); Ptot = 1.0;
    t0 = tic;
    for b = 1:n_sub
        H_b = squeeze(H_all(b, 1:K_, :));
        [SE_k(i, b), TR_k(i, b)] = one_sample(H_b, wr, K_, sigma2, Ptot, Imax);
    end
    fprintf('  K=%2d  SE_end=%.4f true=%.4f  (%.1fs, %s)\n', ...
        K_, mean(SE_k(i,:), 'omitnan'), mean(TR_k(i,:), 'omitnan'), toc(t0), datestr(now, 'HH:MM:SS'));
end

for i = 1:length(P_list)
    K_ = 10; Ptot = P_list(i);
    t0 = tic;
    for b = 1:n_sub
        H_b = squeeze(H_all(b, 1:K_, :));
        [SE_p(i, b), TR_p(i, b)] = one_sample(H_b, wr, K_, sigma2, Ptot, Imax);
    end
    fprintf('  P=%7.4fW (%+ddBW)  SE_end=%.4f true=%.4f  (%.1fs, %s)\n', ...
        Ptot, round(10*log10(Ptot)), mean(SE_p(i,:), 'omitnan'), mean(TR_p(i,:), 'omitnan'), ...
        toc(t0), datestr(now, 'HH:MM:SS'));
end

elapsed = toc(t_all);
fprintf('== sweep done in %.1f s ==\n', elapsed);
save(out_file, 'SE_k', 'TR_k', 'SE_p', 'TR_p', 'K_list', 'P_list', 'n_sub', 'Imax', 'sigma2', 'elapsed');
fprintf('Saved: %s\n', out_file);
end

function [se_end, true_r] = one_sample(H_b, wr, K_, sigma2, Ptot, Imax)
% 单样本官方链：NF_group_1 → NF_NOMA，SE(end) + 返回功率处真 rate
Ht = (H_b * conj(wr)).';                                   % main:169-170
[Hr, F, rf_num, setf] = NF_group_1(Ht, K_, Ht);             % main:171
[SE, ~, ~, pw] = NF_NOMA(Hr, F, setf, K_, rf_num, sigma2, Ptot, Imax);   % main:190
se_end = SE(end);                                          % main:207 口径
true_r = true_rate_at_power(Hr, F, setf, rf_num, pw, sigma2);
end

function r = true_rate_at_power(Hr, F, setf, rf_num, pw, sigma2)
% myfun_NF 的 bet 模型在返回功率处的真实 rate（等价 A,C 取 MMSE 最优）
K_ = size(F, 2);
Fi = F;
for k = 1:K_
    Fi(:, k) = Fi(:, k) / norm(Fi(:, k));    % NF_NOMA:25-28
end
H_eq = Hr' * Fi;                             % NF_NOMA:29-30
ind = zeros(1, rf_num); usel = 0;
for n = 1:rf_num
    usern = setf(n, :); usern(usern == 0) = [];
    ind(n) = usel + length(usern); usel = ind(n);
end
ind = [0, ind];
r = 0;
for n = 1:rf_num
    usern = setf(n, :); usern(usern == 0) = [];
    for m = 1:length(usern)
        u = usern(m);
        p_um = pw(ind(n) + m);
        bet = abs(H_eq(u, u))^2 * sum(pw(ind(n) + 1:ind(n) + m - 1)) + sigma2;   % myfun_NF:21
        for j = 1:rf_num
            if j ~= n
                usern_j = setf(j, :); usern_j(usern_j == 0) = [];
                bet = bet + abs(H_eq(u, usern_j(1)))^2 * sum(pw(ind(j) + 1:ind(j + 1)));  % myfun_NF:24
            end
        end
        r = r + log2(1 + p_um * abs(H_eq(u, u))^2 / bet);
    end
end
end
