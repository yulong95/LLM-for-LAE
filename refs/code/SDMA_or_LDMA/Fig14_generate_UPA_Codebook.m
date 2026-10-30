clear all;
clc;
close all;

c = 3e8;
Nt1 = 256;
Nt2 = 16;
fc = 30e9; % Frequencey 
lambda = c/fc; % wavelegenth;
d = lambda/2;

D = sqrt((Nt1*d)^2+(Nt2*d)^2); % array apterture
D_rayleigh = 2*D^2/lambda; % Rayleigh distance

%% calculate G(beta1) *  G(beta2) and derive r for each angle
delta = 0.55;
rho_min = 4;
[w, theta_list, phi_list, r_list] = Code3D(Nt1, Nt2, d, lambda, delta, rho_min);

correlation = abs(w'*w);
correlation(logical(eye(size(correlation))))=0;
[correlation_max, max_idx] = max(max(correlation));

save('Codebook3d_30G_256_16.mat');