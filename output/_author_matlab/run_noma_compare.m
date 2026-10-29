function run_noma_compare(input_file, output_file)
% run_noma_compare — 对照作者 NF_NOMA 两种调用约定 + 等功率控制
% (a) as-repo:   NF_NOMA(H_b, F_group.', ...)          — 仓库包装器/test_noma 约定
% (b) verified:  NF_NOMA(H_b.', w_sel*F_group, ...)    — NF_NOMA 内部数学自洽约定
%                 H_eq = H'*Fi = G*F，且 NF_group_1 的 ZF 使其对角化
% (c) eq-power:  与 NF_NOMA 相同 SINR 模型 + 等功率 p=P/K（约定 b 下）
% 只调用未改动的作者文件：QuaCode.m / NF_group_1.m / NF_NOMA.m
N = 256;
d = 0.005;
lambda = 0.01;
Imax = 5;
rho_min = 4;
delta = 1.8;

data = load(input_file);
H_batch = data.H_batch;     % [B, K, N] complex
sigma2 = data.sigma2;
P_total = data.P_total;
[B, K, ~] = size(H_batch);

[w_near, ~, ~] = QuaCode(N, d, lambda, delta, rho_min);
w_near_reshape = reshape(w_near, N, []);  % [N, Q]
num_theta = size(w_near, 2);
num_dis = size(w_near, 3);

sum_a = nan(B, 1); sum_b = nan(B, 1); sum_eq = nan(B, 1); sum_d = nan(B, 1);
rf_nums = zeros(B, 1); fail_a = 0; fail_b = 0;

tic;
for b = 1:B
    H_b = squeeze(H_batch(b, :, :));  % [K, N]

    %% beam selection (author select_beam_near logic, inline)
    precoding_matrix_near_near = zeros(N, K);
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
        idx_b = max_idx - num_dis * (idx_a - 1);
        precoding_matrix_near_near(:, i_user) = conj(w_near(:, idx_a, idx_b));
    end

    %% grouping (author NF_group_1)
    H_to_group = H_b * conj(w_near_reshape);  % [K, Q]
    H_to_group = H_to_group.';                 % [Q, K]
    [~, F_group, rf_num, setf] = NF_group_1(H_to_group, K, H_to_group);
    rf_nums(b) = rf_num;

    %% rebuild x2 (retained beams) exactly as NF_group_1 head does
    H_abs = abs(H_to_group);
    setv = zeros(K, 1);
    for k = 1:K
        [~, order] = sort(H_abs(:, k), 'descend');
        setv(k) = order(1);
    end
    [x1, ~] = sort(setv);
    [x2, ~, ~] = unique(x1, 'last');   % sorted retained beam indices

    %% (a) as-repo convention
    try
        Fi_a = F_group.';                       % [K, rf]
        [SE_a, ~, ~, ~] = NF_NOMA(H_b, Fi_a, setf, K, rf_num, sigma2, P_total, Imax);
        sum_a(b) = SE_a(end);
    catch
        fail_a = fail_a + 1;
    end

    %% (b) verified convention: H=[N,K], Fi=[N,K] per-user antenna precoders
    try
        w_sel = w_near_reshape(:, x2);          % [N, rf] codebook columns (no conj)
        Fi_b = w_sel * F_group;                 % [N, K]
        [SE_b, ~, ~, pow_b] = NF_NOMA(H_b.', Fi_b, setf, K, rf_num, sigma2, P_total, Imax);
        sum_b(b) = SE_b(end);

        %% shared SINR model (myfun_NF's bet) helper
        H_eq = (H_b.')' * Fi_b;                 % = conj(H_b)*Fi_b = G*F
        ind = zeros(1, rf_num); usel = 0;
        for n = 1:rf_num
            usern = setf(n, :); usern(usern == 0) = [];
            ind(n) = usel + length(usern); usel = ind(n);
        end
        ind = [0, ind];
        bet_rate = @(pw) local_rate(pw, H_eq, setf, ind, rf_num, sigma2);

        %% (c) equal-power control, same SINR model, p=P/K
        sum_eq(b) = bet_rate(P_total / K * ones(K, 1));

        %% (d) TRUE rate at fmincon's final power (SE(end) uses stale A,C → lower bound)
        sum_d(b) = bet_rate(pow_b(:));
    catch
        fail_b = fail_b + 1;
    end

    if mod(b, 10) == 0
        fprintf('  %d/%d  a=%.4f b=%.4f eq=%.4f true_popt=%.4f rf=%d\n', ...
            b, B, sum_a(b), sum_b(b), sum_eq(b), sum_d(b), rf_num);
    end
end
elapsed = toc;

fprintf('\n===== SUMMARY =====\n');
fprintf('samples=%d  failures: as-repo=%d verified=%d\n', B, fail_a, fail_b);
fprintf('as-repo   SE(end) mean=%.4f\n', mean(sum_a, 'omitnan'));
fprintf('verified  SE(end) mean=%.4f  (WMMSE stale-A,C lower bound)\n', mean(sum_b, 'omitnan'));
fprintf('verified  TRUE rate at fmincon power mean=%.4f\n', mean(sum_d, 'omitnan'));
fprintf('eq-power  TRUE rate mean=%.4f  (same beams/SINR model, p=P/K)\n', mean(sum_eq, 'omitnan'));
fprintf('elapsed=%.1f s (%.3f s/sample)\n', elapsed, elapsed / B);

save(output_file, 'sum_a', 'sum_b', 'sum_eq', 'sum_d', 'rf_nums', 'fail_a', 'fail_b', 'elapsed');
fprintf('Saved: %s\n', output_file);
end

function r = local_rate(pw, H_eq, setf, ind, rf_num, sigma2)
% myfun_NF 的 bet 干扰模型下的直接 SINR rate（功率向量 pw 为 beam-major 堆叠）
    K = size(setf, 2);
    pw = pw(:);
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
