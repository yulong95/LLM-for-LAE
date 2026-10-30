clear all;
clc;
close all;

cd ..
cd cvx
cvx_setup
cd ..
cd LDMA

CoreNum=32;
if isempty(gcp('nocreate'))
    parpool(CoreNum);
end

currentFolder = pwd;
addpath(genpath(currentFolder))

SNR_dB = 20:4:20;
SNR_linear = 10.^(SNR_dB/10.);
N_iter = 192;
Nr = 16;
Nt = 512;

N_user = 10;
N_ris = 1;
Power_total = 1;
SNR = SNR_linear(1);
sigma2 = 1/SNR;

fc = 30e9;
lambda=3e8/fc; % wavelegenth;
d=lambda/2;

L_list = 0:1:10;
num_L = length(L_list);

capacity_near_near_est = zeros(1,num_L);
capacity_near_near_per = zeros(1,num_L);
capacity_near_near_digital = zeros(1,num_L);
capacity_near_near_digital_MMSE = zeros(1,num_L);
capacity_near_near_hybrid_LS = zeros(1,num_L);
capacity_near_near_hybrid_MMSE = zeros(1,num_L);
capacity_near_near_MMSE = zeros(1,num_L);
capacity_near_near_MMSE_est = zeros(1,num_L);

capacity_near_far_est = zeros(1,num_L);
capacity_near_far_per = zeros(1,num_L);
capacity_near_far_MMSE = zeros(1,num_L);
capacity_near_far_MMSE_est = zeros(1,num_L);

for i_L=1:num_L
    L_list(i_L)
    L = L_list(i_L);
    %% generate DFT codebook
    AoD_list = asin(linspace(-1,1-2/Nt,Nt));
    AoA_list = asin(linspace(-1,1-2/Nr,Nr));
    DFT_t = zeros(Nt, Nt);
    DFT_r = zeros(Nr,Nr);

    for l=1:length(AoD_list)
        DFT_t(:,l)=array_respones(AoD_list(l),Nt,d,lambda);
    end
    for l = 1:length(AoA_list)
        DFT_r(:,l)=array_respones(AoA_list(l),Nr,d,lambda);
    end
    
    %% near-field codebook
    rho_min = 4;
    delta = 1.5;
    [w_near, sin_theta_list, r_list] = QuaCode_para(Nt, d, lambda, delta, rho_min);
    num_dis = size(w_near,3);
    num_theta = size(w_near,2);
    
    sum_rate_near_near_est = zeros(1, N_iter);
    sum_rate_near_near_per = zeros(1, N_iter);
    sum_rate_near_near_digital = zeros(1, N_iter);
    sum_rate_near_near_digital_MMSE = zeros(1, N_iter);
    sum_rate_near_near_hybrid_LS = zeros(1, N_iter);
    sum_rate_near_near_hybrid_MMSE = zeros(1, N_iter);
   
    sum_rate_near_near_MMSE = zeros(1, N_iter);
    sum_rate_near_near_MMSE_est = zeros(1, N_iter);
    
    sum_rate_near_far_est = zeros(1, N_iter);
    sum_rate_near_far_per = zeros(1, N_iter);
    sum_rate_near_far_MMSE = zeros(1, N_iter);
    sum_rate_near_far_MMSE_est = zeros(1, N_iter);
    
    parfor i=1:N_iter
        tic();
        %% generate multiple channel for users in the near-field region
        r_circle_max = 100;
        r_circle_min = 4;
        
        sigma_aod = pi/180*5;
        kappa = 1;
        H_mul_user = generate_user_in_circle_multipath(Nt, d, r_circle_min, r_circle_max, fc, N_user, sigma_aod, L, kappa);

        
        %% select the optimal near-field beam
        num_dis = size(w_near,3);
        num_theta = size(w_near,2);
        [precoding_matrix_near_near] = select_beam_near(H_mul_user, w_near, Nt, N_user, num_theta, num_dis);
        
        %% calculate capacity -- near field codebook
        signal_power = 1;
        noise_sigma = signal_power/SNR;
        noise = sqrt(noise_sigma/Nt)*(randn(N_user,Nt)+1i*randn(N_user,Nt))/sqrt(2);
        noise_power = norm(noise(1,:),'fro')^2;
        h_power = norm(H_mul_user(1,:),'fro')^2;

        %% estimated CSI  -- near field ZF
        H_effect_est = (precoding_matrix_near_near.'*(H_mul_user+noise).').';
        F_d = cal_zero_forcing(H_effect_est, precoding_matrix_near_near);
        sum_rate_near_near_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_near*F_d, N_user, sigma2);
        
        %% estimated CSI -- near field WMMSE
        F_d_MMSE = cal_WMMSE(H_effect_est, precoding_matrix_near_near, F_d, N_user, N_user, N_ris, Power_total, sigma2);
        sum_rate_near_near_MMSE_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_near*F_d_MMSE, N_user, sigma2);

        %% fully-digital ZF
        F_d_digital = cal_zero_forcing(H_mul_user+noise, eye(Nt));
        sum_rate_near_near_digital(i) = cal_sum_rate_mu(H_mul_user, F_d_digital, N_user, sigma2);
        
        %% hybrid precoding to approximate fully-digital ZF
        F_hybrid_a_LS = PEAlt(F_d_digital, Nt, N_user, N_user)/sqrt(Nt);
        H_hybrid_LS = (F_hybrid_a_LS.'*(H_mul_user+noise).').';
        
        F_hybrid_d_LS = cal_zero_forcing(H_hybrid_LS, F_hybrid_a_LS);
        sum_rate_near_near_hybrid_LS(i) = cal_sum_rate_mu(H_mul_user, F_hybrid_a_LS*F_hybrid_d_LS, N_user, sigma2);
        
        
        %% calculate capacity -- far field codebook
        precoding_matrix_near_far = select_beam_far(H_mul_user, DFT_t, Nt, N_user);
        
        % add noise
        noise_far = noise;
        H_effect_est_far = (precoding_matrix_near_far.'*(H_mul_user+noise_far).').';
        
        %% estimated CSI  -- far field LS
        F_d_far = cal_zero_forcing(H_effect_est_far, precoding_matrix_near_far);
        sum_rate_near_far_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_far*F_d_far, N_user, sigma2);
        
        %% calculate the precoder with estimated CSI  -- far field WMMSE
        F_d_MMSE_far_est = cal_WMMSE(H_effect_est_far, precoding_matrix_near_far, F_d_far, N_user, N_user, N_ris, Power_total, sigma2);
        sum_rate_near_far_MMSE_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_far*F_d_MMSE_far_est, N_user, sigma2);
        toc();
        i
    end
    capacity_near_near_est(i_L)= mean(sum_rate_near_near_est);
    capacity_near_near_digital(i_L)= mean(sum_rate_near_near_digital);
    capacity_near_near_digital_MMSE(i_L)= mean(sum_rate_near_near_digital_MMSE);
    capacity_near_near_hybrid_LS(i_L)= mean(sum_rate_near_near_hybrid_LS);
    capacity_near_near_hybrid_MMSE(i_L)= mean(sum_rate_near_near_hybrid_MMSE);
    capacity_near_near_per(i_L)= mean(sum_rate_near_near_per);
    capacity_near_near_MMSE(i_L)= mean(sum_rate_near_near_MMSE);
    capacity_near_near_MMSE_est(i_L)= mean(sum_rate_near_near_MMSE_est);
    
    capacity_near_far_est(i_L)= mean(sum_rate_near_far_est);
    capacity_near_far_per(i_L)= mean(sum_rate_near_far_per);
    capacity_near_far_MMSE(i_L)= mean(sum_rate_near_far_MMSE);
    capacity_near_far_MMSE_est(i_L)= mean(sum_rate_near_far_MMSE_est);
end
figure;
hold on
plot(L_list,capacity_near_near_digital,'k-^','Linewidth',1.5);
plot(L_list,capacity_near_near_digital_MMSE,'k--^','Linewidth',1.5);
plot(L_list,capacity_near_near_hybrid_LS,'m-^','Linewidth',1.5);
plot(L_list,capacity_near_near_hybrid_MMSE,'m--','Linewidth',1.5);
plot(L_list,capacity_near_near_MMSE,'r--+','Linewidth',1.5);
plot(L_list,capacity_near_near_MMSE_est,'r-+','Linewidth',1.5);
plot(L_list,capacity_near_near_per,'g--o','Linewidth',1.5);
plot(L_list,capacity_near_near_est,'g-o','Linewidth',1.5);
plot(L_list,capacity_near_far_MMSE,'b--s','Linewidth',1.5);
plot(L_list,capacity_near_far_MMSE_est,'b-s','Linewidth',1.5);
plot(L_list,capacity_near_far_per,'c--d','Linewidth',1.5);
plot(L_list,capacity_near_far_est,'c-d','Linewidth',1.5);
legend('LS: fully-digital','WMMSE: fully-digital','LS: hybrid','WMMSE: hybrid','WMMSE: near-field with perfect CSI','WMMSE: near-field with estimated CSI',...
    'LS: near-field with perfect CSI','LS: near-field with estimated CSI',...
    'WMMSE: far-field with perfect CSI', 'WMMSE: far-field with estimated CSI',...
    'LS: far-field with perfect CSI','LS: far-field with estimated CSI');
xlabel('SNR (dB)');
ylabel('Spectrum Efficiency (bps/Hz)');
grid on;
box on;

save('LDMA_multipath_user10_all_user_kapa1_all_L_0924_1.mat');