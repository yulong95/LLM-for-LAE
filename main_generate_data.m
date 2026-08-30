%% main_generate_data.m
% Generate hybrid-field channel data for LLM-empowered near-field communications
% Reference: "LLM-Empowered Near-Field Communications for Low-Altitude Economy"
%
% Output: Data_user.mat containing:
%   h_near_slant  - (total_samples, N) complex channel matrix (with path loss)
%   index_far_near - (total_samples, 1) far/near field classification (0/1)
%   where total_samples = num_users * K

clear; clc; close all;

%% System Parameters (Table I from paper)
fc = 30e9;                  % Carrier frequency: 30 GHz
c = 3e8;                    % Speed of light
lambda = c / fc;            % Wavelength
d = lambda / 2;             % Antenna spacing (half-wavelength) = 0.5 cm
N = 256;                    % Number of ULA antennas
K = 10;                     % Number of single-antenna users per sample
num_users = 1000;           % Generate 10000 samples (8000 train + 1000 val + 1000 test after split)

% Noise power: sigma^2 = -20 dBW (paper Section V-A and Fig.6-9)
sigma2_dBW = -20;           % dBW
sigma2 = 10^(sigma2_dBW/10); % = 0.01 W

% Transmit power: P = 0 dBW (paper Fig.6-9)
Pt_dBW = 0;                 % dBW
Pt = 10^(Pt_dBW/10);        % = 1 W

% Distance parameters
Deltath = 75;               % RSS threshold distance (m) - near/far field boundary
Rmin = 8.7;                 % Minimum user distance (m) - from Table I
Rmax = 200;                 % Maximum distance (m)
% Note: Rayleigh distance = 2*(N*d)^2/lambda ≈ 1404 m for N=256, d=lambda/2
% All users within 200 m are physically in the near-field (spherical wavefront).
% The hybrid-field model uses Deltath=75m to classify UAV users as near/far for
% the LLM-based precoding task, with different channel characteristics.

% Angular range for LAE scenario (elevation angles)
theta_min = 30 * pi / 180;  % 30 degrees
theta_max = 90 * pi / 180;  % 90 degrees (directly above)

fprintf('System Parameters:\n');
fprintf('  Lambda = %.4f m\n', lambda);
fprintf('  Array aperture = %.2f m (N*d)\n', N*d);
fprintf('  Near-field threshold (Deltath) = %.2f m\n', Deltath);
fprintf('  User distance range: %.1f m - %.1f m\n', Rmin, Rmax);
fprintf('  Generating %d samples with K=%d users each...\n', num_users, K);

%% Generate channel data
total_samples = num_users * K;
h_near_slant = zeros(total_samples, N);      % Channel vectors (complex)
index_far_near = zeros(total_samples, 1);     % Far/near classification (scalar per user)

% ULA element positions (centered)
elem_pos = (-(N-1)/2 : (N-1)/2) * d;  % 1 x N

for idx = 1:num_users
    if mod(idx, 200) == 0
        fprintf('  Processing sample %d / %d ...\n', idx, num_users);
    end

    for k = 1:K
        row = (idx - 1) * K + k;

        % Random user parameters
        theta_k = theta_min + (theta_max - theta_min) * rand;  % angle
        % Distance: 50% near-field, 50% far-field
        if rand < 0.5
            % Near-field user: distance in [Rmin, Deltath]
            r_k = Rmin + (Deltath - Rmin) * rand;
        else
            % Far-field user: distance in [Deltath, Rmax]
            r_k = Deltath + (Rmax - Deltath) * rand;
        end

        % ---- Channel model per paper Eq.2-6 ----
        % Distance from user to n-th element:
        r_n = sqrt(r_k^2 + elem_pos.^2 - 2*r_k*elem_pos*sin(theta_k));

        % Scalar Rayleigh fading (per user, NOT per element)
        % Paper Eq.2-4: alpha_0 is a scalar complex gain per user
        fading = (randn + 1j * randn) / sqrt(2);

        if r_k <= Deltath
            % Near-field: beamfocusing vector b(theta,r) per Eq.5
            b_k = exp(-1j * 2 * pi * (r_n - r_k) / lambda);
            h_k = b_k * fading;
        else
            % Far-field: steering vector a(theta) per Eq.3
            n_idx = (-(N-1)/2 : (N-1)/2);
            a_k = exp(-1j * 2 * pi * d * n_idx * sin(theta_k) / lambda);
            h_k = a_k * fading;
        end

        % Normalize each channel to unit power
        h_k = h_k / norm(h_k);

        % ---- Classification label ----
        % 0 = far-field, 1 = near-field (scalar per user)
        if r_k <= Deltath
            cl_k = 1;
        else
            cl_k = 0;
        end

        % Store
        h_near_slant(row, :) = h_k;
        index_far_near(row, :) = cl_k;
    end
end

%% Save
save_path = fullfile(fileparts(mfilename('fullpath')), 'Data_user.mat');
save(save_path, 'h_near_slant', 'index_far_near');
fprintf('Data saved to: %s\n', save_path);
fprintf('  h_near_slant: [%d x %d] complex\n', size(h_near_slant));
fprintf('  index_far_near: [%d x %d] (scalar per user)\n', size(index_far_near));
fprintf('  Near-field samples: %d (%.1f%%)\n', sum(index_far_near(:,1)), ...
    100*sum(index_far_near(:,1))/total_samples);
fprintf('Done!\n');
