clear all;
clc;
close all;
%% 初始参数设置
SNR_dB = 10:2:30;  
SNR_linear = 10.^(SNR_dB/10.);

M = 16; % 

n = 512;    % number of beams (transmit antennas)
% n = 256;    % number of beams (transmit antennas)
K = 4;     % number of users
% K = 16;     % number of users
max_K = 400; % 

Nr = 16;    % number of beams (receiver antennas)
% Nr = 16;    % number of beams (receiver antennas)

c = 3e8;
fc =30e9;
lamada = c/fc; % wavelength
d = lamada/2;

N = K/2;      % number of RF chains 
% N = K;      % number of RF chains

P = 1;     % total transmitted power

% 迭代次数设置
% N_iter = 20; % 
N_iter = 200; % 
Imax = 30;  % iteration times of power allocation

Power_total = 1;%

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
for i_snr = 1:length(SNR_dB) 
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


    for iter = 1:N_iter
        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
        %% 产生信道
%         r_circle_max = 100;
        r_circle_max = 40;
        r_circle_min = 4;   
        
        sigma_aod = pi/180*5;
        L = 5;
        kappa = 8;
        H_mul_user = generate_user_in_circle_multipath(n, d, r_circle_min, r_circle_max, fc, K , sigma_aod, L, kappa);
        H_mul_user_conjugate = conj(H_mul_user); % 共轭

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


        % NF_group_1（select_beam_near.m和reduce_RF.m结合）
        H_to_group_1 = H_mul_user*conj(w_near_reshape); % K*Q
        H_to_group_1 = H_to_group_1.'; % Q*K
        [Hr3_1,F3_1,rf_num3_1,setf3_1] = NF_group_1(H_to_group_1,K,H_to_group_1);
        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
        index1 = 1 % 指示代码运行进程
        

        H_to_group_4 = H_mul_user*conj(w_near_reshape); % K*Q,与H_to_group_1相同
        H_to_group_4 = H_to_group_4.'; % Q*K
        [Hr3_4,F3_4,setf3_4] = NF_group_4(H_to_group_4,K,N,n,d,r_circle_min,r_circle_max,fc,sigma_aod,L,kappa,w_near_reshape);
        rf_num3_4 = N;

        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
        index4 = 4 % 指示代码运行进程




        %% Near_field_NOMA
        [SE_1,EE_1,ite_1,power1_1] = NF_NOMA(Hr3_1,F3_1,setf3_1,K,rf_num3_1,1/SNR,P,Imax);
        temp3_1 = temp3_1+SE_1;

        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
        index11 = 1 % 指示代码运行进程




                                    
    end
    
    C_digital(i_snr) = temp/N_iter;
    capacity_near_near_digital(i_snr)= mean(sum_rate_near_near_digital);
    capacity_near_far_est(i_snr)= mean(sum_rate_near_far_est);
    capacity_near_near_est(i_snr)= mean(sum_rate_near_near_est);
    C3_1(i_snr) = temp3_1(end)/N_iter;  
        
end



%% 画图 plot
figure;
hold on 
plot(SNR_dB,C_digital,'g-*','Linewidth',1.5); % 绿色
% plot(SNR_dB,capacity_near_near_digital,'r-*','Linewidth',1.5); % 红色
plot(SNR_dB,capacity_near_far_est,'b-*','Linewidth',1.5); % 蓝色
plot(SNR_dB,capacity_near_near_est,'y-*','Linewidth',1.5);% 黄色
plot(SNR_dB,C3_1,'m-d','Linewidth',1.5); % 品红色 菱形
legend('Fully digital system', 'SDMA', 'LDMA', 'Near-field NOMA');
xlabel('SNR (dB)');
ylabel('Spectral efficiency');
grid on;
box on;







