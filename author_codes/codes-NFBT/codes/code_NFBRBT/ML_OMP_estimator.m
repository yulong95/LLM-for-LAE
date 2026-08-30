function [h_hat_ML, h_hat_OMP] = ML_OMP_estimator(h, P, N_ports, L, sigma2, Iteration)
% y is a vector of length N_ports * P.
% S is a matrix of size (P, N).
% L is the number of paths.
N = length(h);

f_c = 2.4e9;        % Frequency
Lambda = 3e8/f_c;   % Wavelength
W = 10;             % Aperture length per wavelength

x = linspace(0,W*Lambda,N)';

% x = Lambda/2*[0:1:N-1];

%% S_Init
S = zeros(P,N);
if P*N_ports<=N
    Index_act = randperm(N,P*N_ports);
%    Index_act = floor(linspace(1,N,P*N_ports));
%    Index_act(randperm(P*N_ports,P*N_ports)) = Index_act;
else
    Index_act = [randperm(N,N),randi(N,1,P*N_ports-N)];
end
for p = 1:P
    S(p,Index_act((p-1)*N_ports + 1:1:p*N_ports)) = 1;
end
%%

[P, N]  = size(S);
N_ports = sum(S(1,:));

% atom model: a(theta) = exp(1i*pi*theta*(0:N-1).'), with theta in range [-1, 1].
S_tilde = zeros(N_ports*P, N);
for idx = 1:P
    indActivated = find(S(idx, :));
    for p = 1:N_ports
        S_tilde((idx-1)*N_ports+p, indActivated(p)) = 1;
    end
end
%% Input_init

y = zeros(P*N_ports,1);
for p = 1:P
    y((p-1)*N_ports+1:1:p*N_ports,1) = S_tilde((p-1)*N_ports + 1:1:p*N_ports,:)*(h + sqrt(sigma2)*(randn(N,1)+1j*randn(N,1))/sqrt(2));
end

%% Training parameter
damping_factor = 0.93;
lambda = 0.0007/norm(y)^2;

%% OMP for initial guess.

DFT_mtx = dftmtx(N);

% DFT_mtx = zeros(N,N);
% for m = 1:N
%     for n = 1:N
%         DFT_mtx(m,n) = exp(-1j*2*pi/Lambda*(x(2)-x(1))*(m-1)*(2*n-2-N)/N);
%     end
% end
% DFT_mtx = DFT_mtx/sqrt(N);

A = S_tilde * DFT_mtx;

support = [];
r = y;

for idx_L = 1:L
    gamma = A'*r;
    [~, idx_max] = max(abs(gamma));
    support = [support, idx_max];
    As = A(:, support);
    xs = As \ y;
    y_hat = As*xs;
    r = y - y_hat;
end
x_hat = zeros(L, 1);
x_hat(support) = xs;

thetas = zeros(1, L);
for ell = 1:L
    thetas(ell) = -mod((support(ell)-1)/N*2+1, 2)-1;
end

%% Setup initial values of the gains and thetas.

% Construct the atom matrix A.
A = zeros(N, L);
for idx = 1:L
    A(:, idx) = exp(1i*pi*thetas(idx)*(0:N-1).');
end

dA = zeros(N, L);
gains = (S_tilde*A)\y;

h_hat_OMP = A*gains;

for iter = 1:Iteration

    for idx = 1:L
        dA(:, idx) = 1i*pi*((0:N-1).') .* exp(1i*pi*(0:N-1).'*thetas(idx));
    end
    r = y-S_tilde*A*gains;

    dLoss_dtheta = zeros(1, L);

    for idx = 1:L
        dLoss_dtheta(idx) = (r')*(-S_tilde*dA(:, idx)*gains(idx));
    end
    thetas = thetas - lambda*2*real(dLoss_dtheta);

    for idx = 1:L
        A(:, idx) = exp(1i*pi*thetas(idx)*(0:N-1).');
    end

    gains = (S_tilde*A)\y;

    r = y - S_tilde*A*gains;
%     fprintf('Iter = %d: Loss = %.4f\n', iter, norm(r)^2);

    lambda = lambda * damping_factor;
end

h_hat_ML = A*gains;

NMSE_ML = mag2db(norm(h_hat_ML - h)/norm(h));
NMSE_OMP = mag2db(norm(h_hat_OMP - h)/norm(h));

if NMSE_ML > NMSE_OMP
    h_hat_ML = h_hat_OMP;
end

end

