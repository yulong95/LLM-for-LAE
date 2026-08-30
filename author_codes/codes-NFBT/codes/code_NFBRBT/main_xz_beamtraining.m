clc;
clear all
close all

N = 256; % the number of the antennas at the BS
K = 1;% the number of users
M = 1;% number of subcarriers
L = 1; % number of paths per user

A = 2;

fc = 100e9; % carrier frequency
Rmin = 4;
Rmax = 4;
sector = pi/6;

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


sample = 20;

SNR_dB = 10:2:30;
SNR_linear = 10.^(SNR_dB/10.);


% generate the far-field codebook 
s = 1;
D = s*N; 
row = (-(N - 1)/2:(N - 1)/2)' ;
col = -1 + 2/D : 2/D : 1 ;
theta_fn = asin(col);
DFT  =  exp( 1j*  pi * row * col ) / sqrt(N);
% generate the near-field codebook
rho = 3;
eta = 1.2;
rho_max = 64;
[QUA_0,label,dict_cell, label_cell] = QuaCode(N, s, d, lambda_c, eta, rho,rho_max);
QUA_pinv = pinv(QUA_0);
QUA = QUA_0';
S = size(QUA, 1);

select_max_theta = zeros(10,11);% for far and near beamtraining



% generate the Kernal_1

Kernal_J0 = zeros(S,S);
Kernal_exp = zeros(S,S);
for aa = 1:S
    for bb = 1:S
        Kernal_J0(aa,bb) = besselj(0,norm(QUA(bb,:)-QUA(aa,:)));
        Kernal_exp(aa,bb) = exp(-norm(QUA(bb,:)-QUA(aa,:))^2);
    end
end



P1 = [1,-1+2/D,64,2];
sampling_interval_1 = [2/D,1];
sampling_interval_2 = sampling_interval_1*A;
[w_hierarchical_1, theta_record_list_hierarchical_1,r_record_list_hierarchical_1] = QuaCode_hierarchical(N, d, lambda_c, P1,sampling_interval_2);
w_hierarchical_1 = w_hierarchical_1';
%%%
rate_far = zeros(sample,length(SNR_dB));
rate_near = zeros(sample,length(SNR_dB));
rate_opt = zeros(sample,length(SNR_dB));
rate_far_and_near = zeros(sample,length(SNR_dB));
rate_near_hierarchical = zeros(sample,length(SNR_dB));
rate_near_GPR_1 = zeros(sample,length(SNR_dB));
rate_near_GPR_2 = zeros(sample,length(SNR_dB));
rate_near_GPR_3 = zeros(sample,length(SNR_dB));

for t = 1:sample
    t   
    [H, hc, r, theta, G] = near_field_channel(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 0);
    H = channel_norm(H);
    k = 1;
    Hsf =  reshape(H(k, :, :), [N, M]);    
    Z = Hsf;

   % generate the Gain_vector
    Gain_vector = QUA*Z;
 
   % generate the Kernal_2
   Rep = 100;
   Kernal_SV_mean = zeros(S,S);
   for rp = 1:Rep
       Kernal_SV_mean = Kernal_SV_mean + Gain_vector*Gain_vector';
   end
   Kernal_SV_mean = Kernal_SV_mean/Rep;


   for s = 1:length(SNR_dB)
      s
    SNR = SNR_linear(s);
    %% far-field beam training 
    array_gain_far = 0;
    for i =1:length(DFT)
        array_gain_far=max(array_gain_far,abs(DFT(i,:)*Z)^2);
    end
    rate_far(t,s) = log2(1 + SNR * array_gain_far);
    %% near-field beam training
    array_gain_near = 0;
    for i =1:size(QUA,1)
         if array_gain_near<=abs(QUA(i,:)*Z)^2
            i_max = i;
            array_gain_near=max(array_gain_near,abs(QUA(i,:)*Z)^2);
         end
    end
    rate_near(t,s) = log2(1 + SNR * array_gain_near);
    %% perfect CSI based beamforming
    wc_opt = exp(1j*angle(Z'))/sqrt(N);
    array_gain = abs(wc_opt*Z)^2;
    rate_opt(t,s) = log2(1 + SNR * array_gain);
    %% Far_and_near_field beam training
    % 1_far
    array_gain_far_and_near_1 = 0;
    for i =1:size(DFT,2)
        if array_gain_far_and_near_1<= abs(Z'*DFT(:,i))^2
           max_index_theta = theta_fn(i);
           array_gain_far_and_near_1 = abs(Z'*DFT(:,i))^2;
        end
    end
    select_max_theta(t,s) = max_index_theta;
    % 2_near
    w_far_and_near = QuaCode_fn(N, s, d, lambda_c, eta, rho,rho_max,max_index_theta);
    w_far_and_near = w_far_and_near';
    array_gain_far_and_near_2 = 0;
    for i =1:size(w_far_and_near,1)
      if array_gain_far_and_near_2<=abs(w_far_and_near(i,:)*Z)^2
         array_gain_far_and_near_2=abs(w_far_and_near(i,:)*Z)^2;
      end
    end
    rate_far_and_near(t,s) = log2(1 + SNR * array_gain_far_and_near_2);  
    %% GPR_based_near_field beam training
    max_GPR_iter = 20;
    [mu_1,cor_1,index_A_1,h_o_1,kmax_1] = GPR_beamtraining(Kernal_J0,SNR,1,S,max_GPR_iter,QUA,Gain_vector);
    mu_baseline = 0;
    for i = 1:S
       if mu_baseline<=abs(mu_1(i))
          mu_baseline=abs(mu_1(i));
          GPR_index_1 = i;
       end
    end
    array_gain_near_GPR_1= abs(QUA(GPR_index_1,:)*Z)^2;
    rate_near_GPR_1(t,s) = log2(1 + SNR * array_gain_near_GPR_1);


    [mu_2,cor_2,index_A_2,h_o_2,kmax_2] = GPR_beamtraining(Kernal_exp,SNR,1,S,max_GPR_iter,QUA,Gain_vector);
    mu_baseline = 0;
    for i = 1:S
       if mu_baseline<=abs(mu_2(i))
          mu_baseline=abs(mu_2(i));
          GPR_index_2 = i;
       end
    end
    array_gain_near_GPR_2= abs(QUA(GPR_index_2,:)*Z)^2;
    rate_near_GPR_2(t,s) = log2(1 + SNR * array_gain_near_GPR_2);



    [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining(Kernal_SV_mean,SNR,1,S,max_GPR_iter,QUA,Gain_vector);
    mu_baseline = 0;
    for i = 1:S
       if mu_baseline<=abs(mu_3(i))
          mu_baseline=abs(mu_3(i));
          GPR_index_3 = i;
       end
    end
    array_gain_near_GPR_3= abs(QUA(GPR_index_3,:)*Z)^2;
    rate_near_GPR_3(t,s) = log2(1 + SNR * array_gain_near_GPR_3);



    %% Hierarchical_near_field beam training
    % The first level
    array_gain_near_hierarchical = 0;
    max_index_hierarchical = -1;
    for i = 1:size(w_hierarchical_1,1)
       if array_gain_near_hierarchical<=abs(w_hierarchical_1(i,:)*Z)^2
         max_index_hierarchical = i;
         max_index_theta_hierarchical = theta_record_list_hierarchical_1(1,i);
         max_index_r_hierarchical = r_record_list_hierarchical_1(1,i);
         array_gain_near_hierarchical = abs(w_hierarchical_1(i,:)*Z)^2;
       end
    end
    % The second level
    P2 = [max_index_theta_hierarchical+sampling_interval_2(1)/2,max_index_theta_hierarchical-sampling_interval_2(1)/2,max_index_r_hierarchical+sampling_interval_2(2)/2,max_index_r_hierarchical-sampling_interval_2(2)/2];
    [w_hierarchical_2, theta_record_list_hierarchical_2,r_record_list_hierarchical_2] = QuaCode_hierarchical(N, d, lambda_c, P2,sampling_interval_1);
    w_hierarchical_2 = w_hierarchical_2';
    for i = 1:size(w_hierarchical_2,1)
       if array_gain_near_hierarchical<=abs(w_hierarchical_2(i,:)*Z)^2
         array_gain_near_hierarchical = abs(w_hierarchical_2(i,:)*Z)^2;
       end
    end
    rate_near_hierarchical(t,s) = log2(1 + SNR * array_gain_near_hierarchical);
   end
end

% figure;
% hold on
% plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near),'b^-','Linewidth',1.6)
% plot(SNR_dB,mean(rate_opt),'k--','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_far_and_near),'cd-','Linewidth',1.6)
% plot(SNR_dB,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_1),'gs-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_2),'gd-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_3),'gp-','Linewidth', 1.6)
% legend('Far-field beam training ','Near-field beam training','Perfect CSI based beamforming','Far and near beam training','Hierarchical near beam training','GPR based near field beam training 1','GPR based near field beam training 2','GPR based near field beam training 3')
% xlabel('SNR (dB)');
% ylabel('Achievable Rate (bis/s/Hz)');
% grid on;
% box on;


figure;
hold on
plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(SNR_dB,mean(rate_near),'b^-','Linewidth',1.6)
plot(SNR_dB,mean(rate_opt),'k--','Linewidth', 1.6)
plot(SNR_dB,mean(rate_far_and_near),'cd-','Linewidth',1.6)
plot(SNR_dB,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
plot(SNR_dB,mean(rate_near_GPR_3),'gp-','Linewidth', 1.6)
legend('Far-field beam training ','Near-field beam training','Perfect CSI based beamforming','Far and near beam training','Hierarchical near beam training','GPR based near field beam training ')
xlabel('SNR (dB)');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;

figure;
hold on
plot(x,abs(mu_3),'ms-', 'Linewidth', 1.6)
grid on;
box on;
