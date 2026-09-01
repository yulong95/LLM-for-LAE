function test_basic()
% Quick test: beam selection + NF_group_1 only (no NOMA optimization)
N = 256; d = 0.005; lambda = 0.01;
rho_min = 4; delta = 1.8;

% Generate codebook
[w_near, ~, ~] = QuaCode(N, d, lambda, delta, rho_min);
w_near_reshape = reshape(w_near, N, []);
num_theta = size(w_near, 2);
num_dis = size(w_near, 3);
fprintf('Codebook: %d angles x %d distances = %d codewords\n', num_theta, num_dis, num_theta*num_dis);

% Load test data
data = load('_test_input.mat');
H_batch = data.H_batch;
K = size(H_batch, 2);
fprintf('Input: %d samples, K=%d, N=%d\n', size(H_batch,1), K, size(H_batch,3));

% Process first sample
H_b = squeeze(H_batch(1, :, :));  % [K, N]
fprintf('H_b: %dx%d, max|H|=%.4f\n', size(H_b,1), size(H_b,2), max(abs(H_b(:))));

% Beam selection
beam_near_mul2 = zeros(K, num_theta * num_dis);
cnt = 1;
for i_row = 1:num_theta
    for j_col = 1:num_dis
        beam_near_mul2(:, cnt) = abs(H_b * conj(w_near(:, i_row, j_col)));
        cnt = cnt + 1;
    end
end
fprintf('Beam gains: %dx%d, max=%.4f\n', size(beam_near_mul2,1), size(beam_near_mul2,2), max(beam_near_mul2(:)));

% Find best beam per user
for i_user = 1:K
    [max_val, max_idx] = max(beam_near_mul2(i_user, :));
    idx_a = ceil(max_idx / num_dis);
    idx_b = max_idx - num_dis * (idx_a - 1);
    fprintf('User %d: best beam=%d (angle=%d, dist=%d), gain=%.4f\n', i_user, max_idx, idx_a, idx_b, max_val);
end

% NF_group_1
H_to_group = H_b * conj(w_near_reshape);  % [K, Q]
H_to_group = H_to_group.';                 % [Q, K]
fprintf('H_to_group: %dx%d\n', size(H_to_group,1), size(H_to_group,2));

try
    [Hr, F_group, rf_num, setf] = NF_group_1(H_to_group, K, H_to_group);
    fprintf('NF_group_1 OK: rf_num=%d\n', rf_num);
    fprintf('setf:\n');
    disp(setf);
catch e
    fprintf('NF_group_1 ERROR: %s\n', e.message);
end

fprintf('Test complete.\n');
end
