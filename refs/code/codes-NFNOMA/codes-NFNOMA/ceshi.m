% 第二版_在第一版的基础上，增加主循环迭代次数，同时注释掉第一次效果不好的方法；
% 第三版_在第二版的基础上，尝试增加Bichai_Wang的分组函数
clear all;
clc;
close all;
%% 初始参数设置
SNR_dB = 10:2:30;  
SNR_linear = 10.^(SNR_dB/10.);

M = 16; % 在NF_group_3中SOMP.m里面用到

n = 512;    % number of beams (transmit antennas)
% 为什么将原LDMA代码中的用户数K=4改为K=32后，LDMA(ZF)在SNR较低的时候会高于全数字？
% K = 4;     % number of users
K = 16;     % number of users
max_K = 400; % 在NF_group_6里面用到

Nr = 16;    % number of beams (receiver antennas)
% Nr = 16;    % number of beams (receiver antennas)

c = 3e8;
fc =30e9;
lamada = c/fc; % wavelength
d = lamada/2;

N = K/2;      % number of RF chains (不采用reduce的时候使用)
% N = K;      % number of RF chains

% P = 32;     % total transmitted power
% 将P设置为1，此时Full digital_1和Full digital_2的曲线重合
P = 1;     % total transmitted power

% 迭代次数设置
% N_iter = 20; % 主循环迭代次数_第一次_20231025_afternoon_to_night
N_iter = 200; % 主循环迭代次数_第二次_20231025_night_to_20231026_morning
Imax = 30;  % iteration times of power allocation

Power_total = 1;%暂时用不上，如果需要比较MMSE方式可能用的上

capacity_near_near_est = zeros(1,length(SNR_linear)); % LDMA迫零
capacity_near_near_per = zeros(1,length(SNR_linear));
capacity_near_near_digital = zeros(1,length(SNR_linear)); % 全数字迫零
capacity_near_near_digital_MMSE = zeros(1,length(SNR_linear));
capacity_near_near_hybrid_LS = zeros(1,length(SNR_linear));
capacity_near_near_hybrid_MMSE = zeros(1,length(SNR_linear));
capacity_near_near_MMSE = zeros(1,length(SNR_linear));
capacity_near_near_MMSE_est = zeros(1,length(SNR_linear));

capacity_near_far_est = zeros(1,length(SNR_linear)); % SDMA迫零
capacity_near_far_per = zeros(1,length(SNR_linear));
capacity_near_far_MMSE = zeros(1,length(SNR_linear));
capacity_near_far_MMSE_est = zeros(1,length(SNR_linear));

capacity_near_NOMA = zeros(1,length(SNR_linear)); % Near-field NOMA


%% 产生远场码本 generate DFT codebook
AoD_list = asin(linspace(-1,1-2/n,n));
AoA_list = asin(linspace(-1,1-2/Nr,Nr));
DFT_t = zeros(n, n);
DFT_r = zeros(Nr,Nr);

for l=1:length(AoD_list)
    DFT_t(:,l)=array_respones(AoD_list(l),n,d,lamada);
end
for l = 1:length(AoA_list)
    DFT_r(:,l)=array_respones(AoA_list(l),Nr,d,lamada);
end


%% 产生近场码本 generate near-field codebook
rho_min = 4;
delta = 1.8;
[w_near, sin_theta_list, r_list] = QuaCode(n, d, lamada, delta, rho_min);
w_near_reshape = reshape(w_near,n,[]); % 对w_near做一个reshape，使其变成(n*Q,其中Q>n)
num_Q = size(w_near_reshape,2); % Q
num_dis = size(w_near,3);
num_theta = size(w_near,2);


%% 主函数循环
i_snr = 1; 
    i_snr % 指示代码运行进程

    SNR = SNR_linear(i_snr);
    sigma2 = 1/SNR;

    temp = 0; temp1 = 0; temp2 = 0; temp3 = zeros(1,Imax); temp4 = zeros(1,Imax); temp5 = 0; temp6 = 0;
    temp22 = 0; temp33 = zeros(1,Imax); temp44 = zeros(1,Imax);  temp55 = 0; temp66 = 0;

    temp3_1 = zeros(1,Imax);
    temp3_2 = zeros(1,Imax);
    temp3_3 = zeros(1,Imax);
    temp3_4 = zeros(1,Imax);
    temp3_5 = zeros(1,Imax); 
    temp3_6 = zeros(1,Imax);
    temp3_7_1 = zeros(1,Imax);
    temp3_7_2 = zeros(1,Imax);
    
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

    sum_rate_near_NOMA = zeros(1, N_iter); % NOMA


    iter = 1;
        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
        %% 产生信道
        r_circle_max = 100;
        r_circle_min = 4;   
        
        sigma_aod = pi/180*5;
        L = 5;
        kappa = 8;
        H_mul_user = generate_user_in_line_multipath(n, d, r_circle_min, r_circle_max, fc, K , sigma_aod, L, kappa); % K*n
        H_mul_user_conjugate = conj(H_mul_user); % 共轭
        H_mul_user_transpose = H_mul_user.'; % 转置（transpose）,n*K

        %% select the optimal near-field beam
        num_dis = size(w_near,3);
        num_theta = size(w_near,2);
        [precoding_matrix_near_near] = select_beam_near(H_mul_user, w_near, n, K, num_theta, num_dis);
        %% calculate capacity -- near field codebook
        signal_power = 1;
        noise_sigma = signal_power/SNR;
        noise = sqrt(noise_sigma/n)*(randn(K,n)+1i*randn(K,n))/sqrt(2);
        noise_power = norm(noise(1,:),'fro')^2;
        h_power = norm(H_mul_user(1,:),'fro')^2;
        %% calculate capacity -- far field codebook
        precoding_matrix_near_far = select_beam_far(H_mul_user, DFT_t, n, K);
        
        % add noise
        noise_far = noise;
        H_effect_est_far = (precoding_matrix_near_far.'*(H_mul_user+noise_far).').';
        
        %% Full digital_1 
        H_beam = H_mul_user';
        F = H_beam*inv(H_beam'*H_beam);
        beta = sqrt(P/trace(F'*F));
        H_eq = H_beam'*F;
        for k = 1:K
            sum_inf = sum(abs(H_eq(k,:)).^2)-abs(H_eq(k,k))^2;
            temp = temp+log2(1+abs(H_eq(k,k))^2/(sum_inf+1/(SNR*beta^2)));
        end
        %% Full digital_2 
        F_d_digital = cal_zero_forcing(H_mul_user+noise, eye(n));
        sum_rate_near_near_digital(iter) = cal_sum_rate_mu(H_mul_user, F_d_digital, K, sigma2);

        %% Far_field_SDMA_ZF
        F_d_far = cal_zero_forcing(H_effect_est_far, precoding_matrix_near_far);
        sum_rate_near_far_est(iter) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_far*F_d_far, K, sigma2);

        %% Near_field_LDMA_ZF
        H_effect_est = (precoding_matrix_near_near.'*(H_mul_user+noise).').';
        F_d = cal_zero_forcing(H_effect_est, precoding_matrix_near_near);
        sum_rate_near_near_est(iter) = cal_sum_rate_mu(H_mul_user, precoding_matrix_near_near*F_d, K, sigma2);

        %% 进行用户分组or运行类似于reduce_RF的函数
        % NF_group_7(结合Bichai_Wang的user_grouping.m函数优化一下用户分组)
        
        [HA_full,AP_full,setu] = A_precoder_NF(H_mul_user_transpose , n , N , K , w_near , num_theta , num_dis); % HA_full:K*N,AP_full:n*N,setu:N*1
        setf3_7 = user_grouping_NF(HA_full,N,K,setu); % N*K
        F3_7 = D_precoder_NF(HA_full,AP_full,K,N,setf3_7); % N*K
        rf_num3_7 = N;
        Hr3_7_1 = HA_full.'; % 因为不确定具体是共轭转置还是非共轭转置，所以两种都试一下 N*K
        Hr3_7_2 = HA_full'; % 因为不确定具体是共轭转置还是非共轭转置，所以两种都试一下 N*K

        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
        index7 = 7 % 指示代码运行进程

       





