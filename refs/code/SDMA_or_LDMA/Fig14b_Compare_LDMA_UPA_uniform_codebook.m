clear all;
clc;
close all;

currentFolder = pwd;
addpath(genpath(currentFolder))

c = 3e8;
Nt1 = 256;
Nt2 = 16;
fc = 30e9;
lambda = c/fc;
d = lambda/2;
D = sqrt((Nt1*d)^2+(Nt2*d)^2);
D_rayleigh = 2*D^2/lambda;
D_fresnel = D/2*(D/lambda)^(1/3);

SNR_dB = 10:2:30;
SNR_linear = 10.^(SNR_dB/10.);
N_iter = 400;

N_user = 4;
N_ris = 1;
Power_total = 1;

capacity_near_near_est = zeros(1,length(SNR_linear));
capacity_near_near_per = zeros(1,length(SNR_linear));
capacity_near_near_digital = zeros(1,length(SNR_linear));
capacity_near_near_digital_MMSE = zeros(1,length(SNR_linear));
capacity_near_near_hybrid_LS = zeros(1,length(SNR_linear));
capacity_near_near_hybrid_MMSE = zeros(1,length(SNR_linear));
capacity_near_near_MMSE = zeros(1,length(SNR_linear));
capacity_near_near_MMSE_est = zeros(1,length(SNR_linear));

capacity_near_far_est = zeros(1,length(SNR_linear));
capacity_near_far_per = zeros(1,length(SNR_linear));
capacity_near_far_MMSE = zeros(1,length(SNR_linear));
capacity_near_far_MMSE_est = zeros(1,length(SNR_linear));

%% generate DFT codebook
[DFT_t, theta_grid_far, phi_grid_far] = far_field_codebook_3D(Nt1, Nt2, d, lambda);

%% near-field codebook
load Codebook3d_30G_256_16_uniform.mat;
w_near = w;
num_dis = size(w_near,4);
num_theta = size(w_near,3);
num_gamma = size(w_near,2);

for i_snr=1:length(SNR_linear)
    SNR_dB(i_snr)
    SNR = SNR_linear(i_snr);
    sigma2 = 1/SNR;
    
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
    
    for i=1:N_iter
        tic();
        %% generate multiple channel for users in the near-field region
        r_max = 4;
        r_min = 50;
        
        sigma_aod = pi/180*5;
        L = 5;
        kappa = 8;
            H_mul_user = generate_user_in_circle_multipath_3D(Nt1, Nt2, d, r_min, r_max, fc, N_user, sigma_aod, L, kappa);
        
        %% select the optimal near-field beam
        [precoding_matrix_near_near] = select_beam_near_3D(H_mul_user, w_near,...
            Nt1, Nt2, N_user, num_gamma, num_theta, num_dis);
        
        %% calculate capacity -- near field codebook
        signal_power = 1;
        noise_sigma = signal_power/SNR;
        noise = sqrt(noise_sigma/Nt1/Nt2)*(randn(N_user,Nt1*Nt2)+1i*randn(N_user,Nt1*Nt2))/sqrt(2);
        noise_power = norm(noise(1,:),'fro')^2;
        h_power = norm(H_mul_user(1,:),'fro')^2;
        
        %% estimated CSI  -- near field ZF
        H_effect_est = (precoding_matrix_near_near.'*(H_mul_user+noise).').';
        F_d = cal_zero_forcing(H_effect_est, precoding_matrix_near_near);
        sum_rate_near_near_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_near*F_d, N_user, sigma2);
        
        %% estimated CSI -- near field WMMSE
        F_d_MMSE = cal_WMMSE(H_effect_est, precoding_matrix_near_near, F_d, N_user, N_user, N_ris, Power_total, sigma2);
        sum_rate_near_near_MMSE_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_near*F_d_MMSE, N_user, sigma2);
        
        %% calculate capacity -- far field codebook
        precoding_matrix_near_far = select_beam_far_3D(H_mul_user, DFT_t, Nt1, Nt2, N_user);
        
        % add noise
        noise_far = noise;
        H_effect_est_far = (precoding_matrix_near_far.'*(H_mul_user+noise_far).').';
        
        %% estimated CSI  -- far field LS
        F_d_far = cal_zero_forcing(H_effect_est_far, precoding_matrix_near_far);
        sum_rate_near_far_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_far*F_d_far, N_user, sigma2);
        
        %% estimated CSI  -- far field WMMSE
        F_d_MMSE_far_est = cal_WMMSE(H_effect_est_far, precoding_matrix_near_far, F_d_far, N_user, N_user, N_ris, Power_total, sigma2);
        sum_rate_near_far_MMSE_est(i) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_far*F_d_MMSE_far_est, N_user, sigma2);
        toc();
        i
    end
    capacity_near_near_est(i_snr)= mean(sum_rate_near_near_est);
    capacity_near_near_digital(i_snr)= mean(sum_rate_near_near_digital);
    capacity_near_near_digital_MMSE(i_snr)= mean(sum_rate_near_near_digital_MMSE);
    capacity_near_near_hybrid_LS(i_snr)= mean(sum_rate_near_near_hybrid_LS);
    capacity_near_near_hybrid_MMSE(i_snr)= mean(sum_rate_near_near_hybrid_MMSE);
    capacity_near_near_per(i_snr)= mean(sum_rate_near_near_per);
    capacity_near_near_MMSE(i_snr)= mean(sum_rate_near_near_MMSE);
    capacity_near_near_MMSE_est(i_snr)= mean(sum_rate_near_near_MMSE_est);
    
    capacity_near_far_est(i_snr)= mean(sum_rate_near_far_est);
    capacity_near_far_per(i_snr)= mean(sum_rate_near_far_per);
    capacity_near_far_MMSE(i_snr)= mean(sum_rate_near_far_MMSE);
    capacity_near_far_MMSE_est(i_snr)= mean(sum_rate_near_far_MMSE_est);
end
figure;
hold on
plot(SNR_dB,capacity_near_near_digital,'k-^','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_digital_MMSE,'k--^','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_hybrid_LS,'m-^','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_hybrid_MMSE,'m--','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_MMSE,'r--+','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_MMSE_est,'r-+','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_per,'g--o','Linewidth',1.5);
plot(SNR_dB,capacity_near_near_est,'g-o','Linewidth',1.5);
plot(SNR_dB,capacity_near_far_MMSE,'b--s','Linewidth',1.5);
plot(SNR_dB,capacity_near_far_MMSE_est,'b-s','Linewidth',1.5);
plot(SNR_dB,capacity_near_far_per,'c--d','Linewidth',1.5);
plot(SNR_dB,capacity_near_far_est,'c-d','Linewidth',1.5);
legend('LS: fully-digital','WMMSE: fully-digital','LS: hybrid','WMMSE: hybrid','WMMSE: near-field with perfect CSI','WMMSE: near-field with estimated CSI',...
    'LS: near-field with perfect CSI','LS: near-field with estimated CSI',...
    'WMMSE: far-field with perfect CSI', 'WMMSE: far-field with estimated CSI',...
    'LS: far-field with perfect CSI','LS: far-field with estimated CSI');
xlabel('SNR (dB)');
ylabel('Spectrum Efficiency (bps/Hz)');
grid on;
box on;

save('LDMA_multipath_user4_UPA_uniform_circle_L5_all_user_kapa8.mat','Nt1','Nt2','SNR_dB','capacity_near_near_digital'...
    ,'capacity_near_near_digital_MMSE','capacity_near_near_hybrid_LS','capacity_near_near_hybrid_MMSE',...
    'capacity_near_near_MMSE','capacity_near_near_MMSE_est','capacity_near_near_per','capacity_near_near_est',...
    'capacity_near_far_MMSE','capacity_near_far_MMSE_est','capacity_near_far_per','capacity_near_far_est');