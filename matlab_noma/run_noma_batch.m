function run_noma_batch(input_file, output_file)
% run_noma_batch — Run author's NF-NOMA on a batch of channels
% Usage: matlab -batch "run_noma_batch('input.mat', 'output.mat')"
%
% Input:  input.mat  — H_batch [B, K, N] complex, sigma2, P_total
% Output: output.mat — sum_rates [B, 1], single_rates [B, K]

%% System parameters
N = 256;
d = 0.005;
lambda = 0.01;
Imax = 5;

%% Load input
data = load(input_file);
H_batch = data.H_batch;     % [B, K, N] complex
sigma2 = data.sigma2;
P_total = data.P_total;
[B, K, ~] = size(H_batch);

%% Generate near-field polar codebook (author's QuaCode)
rho_min = 4;
delta = 1.8;
[w_near, ~, ~] = QuaCode(N, d, lambda, delta, rho_min);
w_near_reshape = reshape(w_near, N, []);  % [N, Q]
num_theta = size(w_near, 2);
num_dis = size(w_near, 3);

%% Process each sample
sum_rates = zeros(B, 1);
single_rates = zeros(B, K);

for b = 1:B
    H_b = squeeze(H_batch(b, :, :));  % [K, N]

    %% Near-field beam selection (author's select_beam_near)
    beam_near_mul2 = zeros(K, num_theta * num_dis);
    cnt = 1;
    for i_row = 1:num_theta
        for j_col = 1:num_dis
            beam_near_mul2(:, cnt) = abs(H_b * conj(w_near(:, i_row, j_col)));
            cnt = cnt + 1;
        end
    end

    precoding_matrix_near_near = zeros(N, K);
    for i_user = 1:K
        [~, max_idx] = max(beam_near_mul2(i_user, :));
        idx_a = ceil(max_idx / num_dis);
        idx_b = max_idx - num_dis * (idx_a - 1);
        precoding_matrix_near_near(:, i_user) = conj(w_near(:, idx_a, idx_b));
    end

    %% User grouping (call author's NF_group_1 directly)
    H_to_group = H_b * conj(w_near_reshape);  % [K, Q]
    H_to_group = H_to_group.';                 % [Q, K]
    [Hr, F_group, rf_num, setf] = NF_group_1(H_to_group, K, H_to_group);

    %% NF-NOMA power optimization (call author's NF_NOMA directly)
    % NF_NOMA expects: H=[N,K] (antenna×user), Fi=[K,rf_num]
    Fi = F_group.';  % [K, rf_num]
    [SE, ~, ~, ~] = NF_NOMA(H_b, Fi, setf, K, rf_num, sigma2, P_total, Imax);
    sum_rates(b) = SE(end);

    if mod(b, 50) == 0
        fprintf('  Processed %d/%d samples, current rate: %.4f\n', b, B, sum_rates(b));
    end
end

%% Save output (single_rates not computed for speed; use sum_rates only)
save(output_file, 'sum_rates');
fprintf('Done. Mean sum-rate: %.4f\n', mean(sum_rates));
end
