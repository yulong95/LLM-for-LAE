function run_noma_official_batch(input_file, output_file)
% 正式评估用：官方 NF-NOMA 调用链（main_near_field_NOMA.m:168-190 同款）逐样本并行（parfor）。
% 算法/设置与 run_noma_official_chain 完全一致（Imax=30, NF_group_1→NF_NOMA），
% 仅并行化样本循环。官方文件零改动。
off = 'C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\codes-NFNOMA\codes-NFNOMA';
cd(off); addpath(off, '-begin');

N = 256; d = 0.005; lam = 0.01; delta = 1.8; rho_min = 4;
Imax = 30;   % 官方 main:32

data = load(input_file);
H_batch = data.H_batch;      % [B,K,N] complex
sigma2 = data.sigma2;
P_total = data.P_total;
[B, K_, ~] = size(H_batch);

[w_near, ~, ~] = QuaCode(N, d, lam, delta, rho_min);
wr = reshape(w_near, N, []);

p = gcp('nocreate');
if isempty(p)
    parpool('local');
end

SE_end = nan(B, 1); true_rate = nan(B, 1);
power_user = nan(B, K_); rf_nums = zeros(B, 1); fail_flag = zeros(B, 1);
tic;
parfor b = 1:B
    H_b = squeeze(H_batch(b, :, :));          % [K,N]，与官方 H_mul_user 同布局
    try
        Ht = (H_b * conj(wr)).';              % main:169-170
        [Hr, F, rf_num, setf] = NF_group_1(Ht, K_, Ht);                    % main:171
        [SE, ~, ~, pw] = NF_NOMA(Hr, F, setf, K_, rf_num, sigma2, P_total, Imax);  % main:190
        SE_end(b) = SE(end);                  % main:207 报告口径
        [tr, pu] = true_rate_at_power(Hr, F, setf, rf_num, pw, sigma2);
        true_rate(b) = tr;
        power_user(b, :) = pu;
        rf_nums(b) = rf_num;
    catch
        fail_flag(b) = 1;
    end
end
elapsed = toc;
fail = sum(fail_flag);
fprintf('official batch: B=%d fail=%d elapsed=%.1fs  SE_end mean=%.4f true mean=%.4f\n', ...
    B, fail, elapsed, mean(SE_end, 'omitnan'), mean(true_rate, 'omitnan'));
save(output_file, 'SE_end', 'true_rate', 'power_user', 'rf_nums', 'fail', 'elapsed');
end

function [r, p_user] = true_rate_at_power(Hr, F, setf, rf_num, pw, sigma2)
% myfun_NF 的 bet 模型在 fmincon 返回功率处的真实 rate（等价 A,C 取 MMSE 最优）
K_ = size(F, 2);
Fi = F;
for k = 1:K_
    Fi(:, k) = Fi(:, k) / norm(Fi(:, k));    % NF_NOMA:25-28 同款列归一
end
H_eq = Hr' * Fi;                             % NF_NOMA:29-30 同款
ind = zeros(1, rf_num); usel = 0;
for n = 1:rf_num
    usern = setf(n, :); usern(usern == 0) = [];
    ind(n) = usel + length(usern); usel = ind(n);
end
ind = [0, ind];
p_user = zeros(1, K_);
r = 0;
for n = 1:rf_num
    usern = setf(n, :); usern(usern == 0) = [];
    for m = 1:length(usern)
        u = usern(m);
        p_um = pw(ind(n) + m);
        p_user(u) = p_um;
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
