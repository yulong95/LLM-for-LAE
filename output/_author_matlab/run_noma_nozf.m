function run_noma_nozf(input_file, output_file)
% run_noma_nozf — 无 ZF 变体对照：Fi = 逐用户最优码本列（不经过 NF_group_1 的 ZF）
% 目的：在干扰受限波束下检验 fmincon 功率优化是否有增益（ZF 波束下已测得 ≈0）
% (e1) NF_NOMA SE(end)（旧 A,C 下界）
% (e2) fmincon 最终功率的真 rate
% (e3) 同波束/SINR 模型等功率真 rate
N = 256; d = 0.005; lambda = 0.01; Imax = 5; rho_min = 4; delta = 1.8;
data = load(input_file);
H_batch = data.H_batch; sigma2 = data.sigma2; P_total = data.P_total;
[B, K, ~] = size(H_batch);

[w_near, ~, ~] = QuaCode(N, d, lambda, delta, rho_min);
w_near_reshape = reshape(w_near, N, []);
num_theta = size(w_near, 2); num_dis = size(w_near, 3);

sum_e1 = nan(B, 1); sum_e2 = nan(B, 1); sum_e3 = nan(B, 1); fail = 0;
tic;
for b = 1:B
    H_b = squeeze(H_batch(b, :, :));  % [K, N]
    try
        %% per-user best codeword (author select_beam_near logic)
        Fi = zeros(N, K);
        for i_user = 1:K
            gains = zeros(1, num_theta * num_dis);
            cnt = 1;
            for i_row = 1:num_theta
                for j_col = 1:num_dis
                    gains(cnt) = abs(H_b(i_user, :) * conj(w_near(:, i_row, j_col)));
                    cnt = cnt + 1;
                end
            end
            [~, max_idx] = max(gains);
            idx_a = ceil(max_idx / num_dis);
            Fi(:, i_user) = w_near(:, idx_a, max_idx - num_dis * (idx_a - 1));
        end
        %% 每用户独立成组（无共享、无 ZF）：beam n ↔ user n
        setf = (1:K).'; rf_num = K;

        [SE, ~, ~, pw] = NF_NOMA(H_b.', Fi, setf, K, rf_num, sigma2, P_total, Imax);
        sum_e1(b) = SE(end);

        H_eq = (H_b.')' * Fi;
        ind = 0:K;  % 每 beam 1 用户
        sum_e2(b) = local_rate2(pw(:), H_eq, setf, ind, rf_num, sigma2);
        sum_e3(b) = local_rate2(P_total / K * ones(K, 1), H_eq, setf, ind, rf_num, sigma2);
    catch
        fail = fail + 1;
    end
    if mod(b, 10) == 0
        fprintf('  %d/%d  SE=%.4f true=%.4f eq=%.4f\n', b, B, sum_e1(b), sum_e2(b), sum_e3(b));
    end
end
elapsed = toc;
fprintf('\n===== NO-ZF SUMMARY =====\n');
fprintf('samples=%d fail=%d\n', B, fail);
fprintf('SE(end) bound mean=%.4f\n', mean(sum_e1, 'omitnan'));
fprintf('true @fmincon power mean=%.4f\n', mean(sum_e2, 'omitnan'));
fprintf('true equal-power mean=%.4f\n', mean(sum_e3, 'omitnan'));
fprintf('power-optimization gain = %.4f\n', mean(sum_e2 - sum_e3, 'omitnan'));
fprintf('elapsed=%.1f s\n', elapsed);
save(output_file, 'sum_e1', 'sum_e2', 'sum_e3', 'fail', 'elapsed');
end

function r = local_rate2(pw, H_eq, setf, ind, rf_num, sigma2)
    r = 0;
    for n = 1:rf_num
        usern = setf(n, :); usern(usern == 0) = [];
        for m = 1:length(usern)
            u = usern(m);
            p_um = pw(ind(n) + m);
            bet = abs(H_eq(u, u))^2 * sum(pw(ind(n) + 1:ind(n) + m - 1)) + sigma2;
            for j = 1:rf_num
                if j ~= n
                    usern_j = setf(j, :); usern_j(usern_j == 0) = [];
                    bet = bet + abs(H_eq(u, usern_j(1)))^2 * sum(pw(ind(j) + 1:ind(j + 1)));
                end
            end
            r = r + log2(1 + abs(H_eq(u, u))^2 * p_um / bet);
        end
    end
end
