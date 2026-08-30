%% Version 7
%% change the eta,eta = 1.2
%% Compare to the version 4, find new problems

clc;
clear all
close all

N = 256; % the number of the antennas at the BS
K = 1;% the number of users
M = 1;% number of subcarriers
L = 1; % number of paths per user

A = 2;

fc = 100e9; % carrier frequency
Rmin = 3;
Rmax = 30;
sector = pi/3;

fs = 100e6; % bandwidth
Q = 64;  % number of pilot blocks
tmax = 20e-9; % maximum of the path delay
f = zeros(1,M);
for m = 1:M
    f(m)=fc+fs/(M)*(m-1-(M-1)/2);
end
c = 3e8;
lambda_c = c/fc;
d = lambda_c / 2;
eps = 1e-3;


alpha_index = linspace(1,100,100);
alpha_length = length(alpha_index);


sample = 20;

SNR_dB = 10:2:30;
SNR_linear = 10.^(SNR_dB/10.);


% generate the far-field codebook 
disp('generate the far-field codebook')
s = 1;
D = s*N; 
row = (-(N - 1)/2:(N - 1)/2)' ;
col = -1 + 2/D : 2/D : 1 ;
theta_fn = asin(col);
DFT  =  exp( 1j*  pi * row * col ) / sqrt(N);
disp('the far-field codebook has been generated')
% generate the near-field codebook
disp('generate the near-field codebook')
rho = 3;
eta = 1.2;
rho_max = 64;
[QUA_0,label,dict_cell, label_cell] = QuaCode(N, s, d, lambda_c, eta, rho,rho_max);
QUA_pinv = pinv(QUA_0);
QUA = QUA_0';
S = size(QUA, 1);
disp('the near-field codebook has been generated')
select_max_theta = zeros(10,11);% for far and near beamtraining



disp('generate the near-field hierarchical codebook')
P1 = [1,-1+2/D,64,2];
sampling_interval_1 = [2/D,1];
sampling_interval_2 = sampling_interval_1*A;
[w_hierarchical_1, theta_record_list_hierarchical_1,r_record_list_hierarchical_1] = QuaCode_hierarchical(N, d, lambda_c, P1,sampling_interval_2);
w_hierarchical_1 = w_hierarchical_1';
disp('the near-field hieerarchical codebook has been generated')

rate_far = zeros(sample,1);
rate_near = zeros(sample,1);
rate_opt = zeros(sample,1);
rate_far_and_near = zeros(sample,1);
rate_near_hierarchical = zeros(sample,1);
rate_near_GPR_3 = zeros(sample,1);

%% generate the Kernal
   disp('generate the Kernal')
   Rep = 1000;
   Kernal_SV_mean = zeros(S,S);
   for rp = 1:Rep
       [H, hc, r, theta, G] = near_field_channel(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 1);
       H = channel_norm(H);
       k = 1;
       Hsf =  reshape(H(k, :, :), [N, M]);    
       Z = Hsf;
       Gain_vector = QUA*Z;
       Kernal_SV_mean = Kernal_SV_mean + Gain_vector*Gain_vector';
   end
   Kernal_SV_mean = Kernal_SV_mean/Rep;
   disp('the Kernal has been generated')

   H_index = zeros(N,sample);
   theta_H_index = zeros(sample,1);
   r_H_index = zeros(sample,1);


%% training


for t = 1:sample
    t   
    [H, hc, r, theta, G] = near_field_channel(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 1);
    H = channel_norm(H);
    k = 1;
    Hsf =  reshape(H(k, :, :), [N, M]);    
    Z = Hsf;

   % generate the Gain_vector
    Gain_vector = QUA*Z;

    Gain_vector = Gain_vector/sqrt(mean(diag(Kernal_SV_mean)));


    SNR = SNR_linear(8);
    %% far-field beam training 
    array_gain_far = 0;
    for i =1:length(DFT)
        array_gain_far=max(array_gain_far,abs(DFT(i,:)*Z)^2);
    end
    rate_far(t,1) = log2(1 + SNR * array_gain_far);
    %% near-field beam training
    array_gain_near = 0;
    for i =1:size(QUA,1)
         if array_gain_near<=abs(QUA(i,:)*Z)^2
            i_max = i;
            array_gain_near=max(array_gain_near,abs(QUA(i,:)*Z)^2);
         end
    end
    rate_near(t,1) = log2(1 + SNR * array_gain_near);
    %% perfect CSI based beamforming
    wc_opt = exp(1j*angle(Z'))/sqrt(N);
    array_gain = abs(wc_opt*Z)^2;
    rate_opt(t,1) = log2(1 + SNR * array_gain);
 
    %% GPR_based_near_field beam training
    max_GPR_iter = 40;
    [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_5(Kernal_SV_mean,SNR,1,S,max_GPR_iter,QUA,Gain_vector);
   end



%% plot
% figure;
% hold on
% plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near),'b^-','Linewidth',1.6)
% plot(SNR_dB,mean(rate_opt),'k--','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_far_and_near),'cd-','Linewidth',1.6)
% plot(SNR_dB,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_3),'gp-','Linewidth', 1.6)
% legend('Far-field beam training ','Near-field beam training','Perfect CSI based beamforming','Far and near beam training','Hierarchical near beam training','GPR based near field beam training ')
% xlabel('SNR (dB)');
% ylabel('Achievable Rate (bis/s/Hz)');
% grid on;
% box on;


% x = zeros(S,1);
% for xx = 1:S
%     x(xx) = xx;
% end
% 


figure;
hold on
plot(x,abs(mu_3),'ms--', 'Linewidth', 1.6)
grid on;
box on;

figure;
hold on
plot(x,abs(Gain_vector),'cs-', 'Linewidth', 1.6)
grid on;
box on;

% 
% 
% figure;
% hold on
% plot(x,8170.56*abs(mu_3),'ms--', 'Linewidth', 1.6)
% plot(x,abs(Gain_vector),'cs-', 'Linewidth', 1.6)
% grid on;
% box on;







