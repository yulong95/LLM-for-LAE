function test_noma2()
% Detailed test: check power, H_eq, and SINR
N = 256; d = 0.005; lambda = 0.01;
rho_min = 4; delta = 1.8;
Imax = 3;
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
H_to_group = H_b * conj(w_near_reshape);  % [K, Q]
H_to_group = H_to_group.';                 % [Q, K]
[Hr, F_group, rf_num, setf] = NF_group_1(H_to_group, K, H_to_group);
Fi = F_group.';

fprintf('=== Before NF_NOMA ===\n');
fprintf('Fi norm per column: ');
for k = 1:K
    fprintf('%.4f ', norm(Fi(:,k)));
end
fprintf('\n');

% Normalize like NF_NOMA does
Fi_norm = Fi;
for k = 1:K
    Fi_norm(:,k) = Fi(:,k) / norm(Fi(:,k));
end
H_eq = H_b' * Fi_norm;
fprintf('H_eq diag (after norm): ');
for k = 1:K
    fprintf('%.4f ', abs(H_eq(k,k))^2);
end
fprintf('\n');

% Call NF_NOMA
[SE, ~, ~, power] = NF_NOMA(H_b, Fi, setf, K, rf_num, sigma2, P_total, Imax);
fprintf('\n=== After NF_NOMA ===\n');
fprintf('SE history: ');
fprintf('%.4f ', SE);
fprintf('\n');
fprintf('Final power (from fmincon): ');
fprintf('%.6f ', power);
fprintf('\n');
fprintf('Total power: %.6f (P_total=%.1f)\n', sum(power), P_total);

% Compute true sum rate with SIC using optimized power
Fi_norm2 = Fi;
for k = 1:K
    Fi_norm2(:,k) = Fi(:,k) / norm(Fi(:,k));
end
H_eq2 = H_b' * Fi_norm2;
fprintf('H_eq diag (recomputed): ');
for k = 1:K
    fprintf('%.4f ', abs(H_eq2(k,k))^2);
end
fprintf('\n');

% Manual SINR computation
rate_sum = 0;
for k = 1:K
    signal = abs(H_eq2(k,k))^2 * power(k);
    interf = 0;
    for j = 1:K
        if j ~= k
            interf = interf + abs(H_eq2(k,j))^2 * power(j);
        end
    end
    sinr = signal / (interf + sigma2);
    r_k = log2(1 + sinr);
    rate_sum = rate_sum + r_k;
    fprintf('User %d: signal=%.6f interf=%.6f sinr=%.4f rate=%.4f\n', k, signal, interf, sinr, r_k);
end
fprintf('Manual sum rate: %.4f\n', rate_sum);
end
