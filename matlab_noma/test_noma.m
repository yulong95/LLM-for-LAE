function test_noma()
% Test NF_NOMA call with proper dimensions
N = 256; d = 0.005; lambda = 0.01;
rho_min = 4; delta = 1.8;
Imax = 3;
sigma2 = 0.01;
P_total = 1.0;

% Generate codebook
[w_near, ~, ~] = QuaCode(N, d, lambda, delta, rho_min);
w_near_reshape = reshape(w_near, N, []);
num_theta = size(w_near, 2);
num_dis = size(w_near, 3);

% Load 1 sample
data = load('_test_input.mat');
H_b = squeeze(data.H_batch(1, :, :));  % [K=10, N=256]
K = size(H_b, 1);

% Beam selection
beam_near_mul2 = zeros(K, num_theta * num_dis);
cnt = 1;
for i_row = 1:num_theta
    for j_col = 1:num_dis
        beam_near_mul2(:, cnt) = abs(H_b * conj(w_near(:, i_row, j_col)));
        cnt = cnt + 1;
    end
end

% NF_group_1
H_to_group = H_b * conj(w_near_reshape);
H_to_group = H_to_group.';
[Hr, F_group, rf_num, setf] = NF_group_1(H_to_group, K, H_to_group);
fprintf('rf_num=%d\n', rf_num);

% Prepare Fi for NF_NOMA
Fi = F_group.';  % [K, rf_num]
fprintf('Fi: %dx%d\n', size(Fi,1), size(Fi,2));
fprintf('H_b: %dx%d (will transpose to %dx%d for NF_NOMA)\n', size(H_b,1), size(H_b,2), size(H_b,2), size(H_b,1));

% Call NF_NOMA
try
    [SE, EE, ite, power] = NF_NOMA(H_b, Fi, setf, K, rf_num, sigma2, P_total, Imax);
    fprintf('NF_NOMA OK!\n');
    fprintf('SE final: %.4f\n', SE(end));
    fprintf('SE history: ');
    fprintf('%.4f ', SE);
    fprintf('\n');
catch e
    fprintf('NF_NOMA ERROR: %s\n', e.message);
    fprintf('Stack: %s\n', e.stack(1).name);
end
end
