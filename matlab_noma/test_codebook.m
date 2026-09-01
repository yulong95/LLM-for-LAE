function test_codebook()
% Compare beam gains between author's codebook and our uniform codebook
N = 256; d = 0.005; lambda = 0.01;
rho_min = 4; delta = 1.8;

% Author's codebook (QuaCode)
[w_near, sin_theta_list, r_list] = QuaCode(N, d, lambda, delta, rho_min);
fprintf('Author codebook: %d angles x %d distances = %d codewords\n', ...
    size(w_near,2), size(w_near,3), size(w_near,2)*size(w_near,3));
fprintf('  angles (sin_theta): %.4f to %.4f\n', min(sin_theta_list), max(sin_theta_list));
fprintf('  distances: %.1f to %.1f m\n', min(r_list), max(r_list));

% Our uniform codebook
n_theta = 60; n_r = 60;
thetas_ours = linspace(-95, 85, n_theta) * pi / 180;
rs_ours = linspace(15, 201, n_r);
fprintf('\nOur codebook: %d angles x %d distances = %d codewords\n', n_theta, n_r, n_theta*n_r);
fprintf('  angles: %.1f to %.1f deg\n', min(thetas_ours)*180/pi, max(thetas_ours)*180/pi);
fprintf('  distances: %.1f to %.1f m\n', min(rs_ours), max(rs_ours));

% Load 1 sample
data = load('_test_input.mat');
H_b = squeeze(data.H_batch(1, :, :));
K = size(H_b, 1);
fprintf('\nH_b: %dx%d, max|H|=%.4f, mean|H|=%.4f\n', size(H_b,1), size(H_b,2), max(abs(H_b(:))), mean(abs(H_b(:))));

% Compute max beam gain with author's codebook
max_gain_author = zeros(K, 1);
for k = 1:K
    gains = zeros(size(w_near,2), size(w_near,3));
    for i = 1:size(w_near,2)
        for j = 1:size(w_near,3)
            gains(i,j) = abs(H_b(k,:) * conj(w_near(:,i,j)))^2;
        end
    end
    max_gain_author(k) = max(gains(:));
end
fprintf('\nMax beam gain (author codebook): ');
fprintf('%.4f ', max_gain_author);
fprintf('\nMean: %.4f\n', mean(max_gain_author));

% Compute max beam gain with our codebook
max_gain_ours = zeros(K, 1);
for k = 1:K
    gains = zeros(n_theta, n_r);
    for i = 1:n_theta
        for j = 1:n_r
            theta = thetas_ours(i);
            r = rs_ours(j);
            n_arr = (0:N-1)';
            x_n = (n_arr - (N-1)/2) * lambda/2;
            r_n = sqrt(r^2 + x_n.^2 - 2*r*x_n*sin(theta));
            b = exp(-1j*2*pi*(r_n - r)/lambda) / sqrt(N);
            gains(i,j) = abs(H_b(k,:) * conj(b))^2;
        end
    end
    max_gain_ours(k) = max(gains(:));
end
fprintf('Max beam gain (our codebook):    ');
fprintf('%.4f ', max_gain_ours);
fprintf('\nMean: %.4f\n', mean(max_gain_ours));

fprintf('\nRatio (author/ours): ');
fprintf('%.2f ', max_gain_author./max_gain_ours);
fprintf('\n');
end
