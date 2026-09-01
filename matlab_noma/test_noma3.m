function test_noma3()
% Test: compare with and without column normalization
N = 256; d = 0.005; lambda = 0.01;
rho_min = 4; delta = 1.8;
Imax = 5;
sigma2 = 0.01;
P_total = 1.0;

[w_near, ~, ~] = QuaCode(N, d, lambda, delta, rho_min);
w_near_reshape = reshape(w_near, N, []);
num_theta = size(w_near, 2);
num_dis = size(w_near, 3);

data = load('_test_input.mat');
H_b = squeeze(data.H_batch(1, :, :));
K = size(H_b, 1);

% Beam selection + grouping
beam_near_mul2 = zeros(K, num_theta * num_dis);
cnt = 1;
for i_row = 1:num_theta
    for j_col = 1:num_dis
        beam_near_mul2(:, cnt) = abs(H_b * conj(w_near(:, i_row, j_col)));
        cnt = cnt + 1;
    end
end
H_to_group = H_b * conj(w_near_reshape);
H_to_group = H_to_group.';
[Hr, F_group, rf_num, setf] = NF_group_1(H_to_group, K, H_to_group);
Fi = F_group.';

fprintf('rf_num=%d\n', rf_num);
fprintf('Fi col norms: ');
for k = 1:K
    fprintf('%.4f ', norm(Fi(:,k)));
end
fprintf('\n');

% Method 1: NF_NOMA as-is (normalizes internally)
fprintf('\n=== Method 1: NF_NOMA as-is ===\n');
[SE1, ~, ~, power1] = NF_NOMA(H_b, Fi, setf, K, rf_num, sigma2, P_total, Imax);
fprintf('SE: %.4f\n', SE1(end));

% Method 2: Compute rate manually with the SAME normalization as NF_NOMA
fprintf('\n=== Method 2: Manual with NF_NOMA normalization ===\n');
Fi_norm = Fi;
for k = 1:K
    Fi_norm(:,k) = Fi(:,k) / norm(Fi(:,k));
end
H_eq = H_b' * Fi_norm;
rate_sum = 0;
for k = 1:K
    signal = abs(H_eq(k,k))^2 * power1(k);
    interf = 0;
    for j = 1:K
        if j ~= k
            interf = interf + abs(H_eq(k,j))^2 * power1(j);
        end
    end
    sinr = signal / (interf + sigma2);
    rate_sum = rate_sum + log2(1 + sinr);
end
fprintf('Sum rate (manual, normalized): %.4f\n', rate_sum);

% Method 3: Compute rate WITHOUT column normalization (use raw Fi)
fprintf('\n=== Method 3: Manual without column normalization ===\n');
H_eq_raw = H_b' * Fi;
% Scale power by ||fi_k||^2 to get equivalent power on raw Fi
power_raw = zeros(K, 1);
for k = 1:K
    power_raw(k) = power1(k) * norm(Fi(:,k))^2;
end
% Re-normalize total power
power_raw = power_raw / sum(power_raw) * P_total;

rate_sum_raw = 0;
for k = 1:K
    signal = abs(H_eq_raw(k,k))^2 * power_raw(k);
    interf = 0;
    for j = 1:K
        if j ~= k
            interf = interf + abs(H_eq_raw(k,j))^2 * power_raw(j);
        end
    end
    sinr = signal / (interf + sigma2);
    rate_sum_raw = rate_sum_raw + log2(1 + sinr);
end
fprintf('Sum rate (manual, raw Fi): %.4f\n', rate_sum_raw);

% Method 4: Direct rate with raw Fi, equal power (no NOMA optimization)
fprintf('\n=== Method 4: Raw Fi, equal power ===\n');
Pk = P_total / K;
rate_eq = 0;
for k = 1:K
    signal = abs(H_eq_raw(k,k))^2 * Pk;
    interf = 0;
    for j = 1:K
        if j ~= k
            interf = interf + abs(H_eq_raw(k,j))^2 * Pk;
        end
    end
    sinr = signal / (interf + sigma2);
    rate_eq = rate_eq + log2(1 + sinr);
end
fprintf('Sum rate (equal power): %.4f\n', rate_eq);
end
