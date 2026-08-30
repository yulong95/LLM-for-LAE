clc;

rng(100);

f_c = 2.4e9;        % Frequency

Lambda = 3e8/f_c;   % Wavelength

load('Kernal.mat');

% h = U*S^(1/2)*(randn(N,1)+1j*randn(N,1))/sqrt(2);

h = CDL_channel(x,f_c,Lambda);

h = h/sqrt(mean(diag(Kernal_CDL_mean)));

% h = h/norm(h);

% h = SV_channel(x,Lambda,L);

sigma2_dB = -20;

sigma2 = 10.^(sigma2_dB/10);   % Noise power

beta = 3;

[h_hat_GPR,cor,index_A,h_o] = GPR_multi(x,h,Kernal_CDL_mean,sigma2,P*N_ports,1);
% [h_hat_ML, h_hat_OMP] = ML_OMP_estimator(h, P, N_ports, L, sigma2, 10);
% [h_hat_GPR_rand,cor,index_A,h_o] = GPR_rand(x,h,Kernal_CDL_mean,sigma2,P*N_ports);


NMSE_GPR = mag2db(norm(h_hat_GPR - h)/norm(h))
% NMSE_OMP = mag2db(norm(h_hat_OMP - h)/norm(h))
% NMSE_ML = mag2db(norm(h_hat_ML - h)/norm(h))
% NMSE_GPR_rand = mag2db(norm(h_hat_GPR_rand - h)/norm(h))

% Sigma = Kernal_CDL_mean;
% 
% NMSE_LowerBound = Kernal_CDL_mean*(Sigma*(Sigma+sigma2*eye(N))^(-1)-eye(N))^2 ...
%     + sigma2 * (Sigma*(Sigma+sigma2*eye(N))^(-1))^2;
% 
% NMSE_LowerBound = trace(NMSE_LowerBound)/N/trace(Sigma);

Plotfigure_2;



