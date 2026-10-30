clear all;
clc;
close all;
%% 初始参数设置
SNR_dB = 20;  
SNR_linear = 10.^(SNR_dB/10.);
SNR = SNR_linear;

M = 16; % 在NF_group_3中SOMP.m里面用到

n = 512;    % number of beams (transmit antennas)
% 为什么将原LDMA代码中的用户数K=4改为K=32后，LDMA(ZF)在SNR较低的时候会高于全数字？
% K = 4;     % number of users
% K = 16;     % number of users
max_K = 400; % 在NF_group_6里面用到

Nr = 16;    % number of beams (receiver antennas)
% Nr = 16;    % number of beams (receiver antennas)

c = 3e8;
fc =30e9;
lamada = c/fc; % wavelength
d = lamada/2;

% N = K/2;      % number of RF chains (不采用reduce的时候使用)
% N = K;      % number of RF chains

% P = 32;     % total transmitted power
% 将P设置为1，此时Full digital_1和Full digital_2的曲线重合
P = 1;     % total transmitted power

% 迭代次数设置
N_iter = 200; % 主循环迭代次数
% N_iter = 100; % 主循环迭代次数
Imax = 30;  % iteration times of power allocation

Power_total = 1;%暂时用不上，如果需要比较MMSE方式可能用的上


K_list = [4,8,12,16,20];
% K_list = [8,16,32,64,128];
capacity_near_near_est = zeros(1,length(K_list)); % LDMA迫零
capacity_near_near_digital = zeros(1,length(K_list)); % 全数字迫零

capacity_near_far_est = zeros(1,length(K_list)); % SDMA迫零



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
for i_snr = 1:length(K_list) 
    i_snr % 指示代码运行进程

    K = K_list(i_snr);
    N = K/2;
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
        r_circle_max = 100;
        r_circle_min = 4;   
        
        sigma_aod = pi/180*5;
        L = 5;
        kappa = 8;
        H_mul_user = generate_user_in_line_multipath(n, d, r_circle_min, r_circle_max, fc, K , sigma_aod, L, kappa);
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
        % NF_group(经过了select_beam_near.m,每个用户选出了一个最大beam)


        % NF_group_1（select_beam_near.m和reduce_RF.m结合）
        H_to_group_1 = H_mul_user*conj(w_near_reshape); % K*Q
        H_to_group_1 = H_to_group_1.'; % Q*K
        [Hr3_1,F3_1,rf_num3_1,setf3_1] = NF_group_1(H_to_group_1,K,H_to_group_1);
        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
%         index1 = 1 % 指示代码运行进程
        
        % NF_group_2(通过伪逆得到近场信道的极域表示)
%         [Hr3,F3,rf_num3,setf3] = NF_group_2(H_to_group,K,H_to_group,N);
        

        % NF_group_3(借鉴崔哥近场信道估计文章的极域表示，跟NF_group_2方法做一个比较)



        % NF_group_4(参考Ruicheng_Jiao的select函数)
        H_to_group_4 = H_mul_user*conj(w_near_reshape); % K*Q,与H_to_group_1相同
        H_to_group_4 = H_to_group_4.'; % Q*K
        [Hr3_4,F3_4,setf3_4] = NF_group_4(H_to_group_4,K,N,n,d,r_circle_min,r_circle_max,fc,sigma_aod,L,kappa,w_near_reshape);
        rf_num3_4 = N;

        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
%         index4 = 4 % 指示代码运行进程
        
        % NF_group_5(reduce_RF.m和select.m结合,即不采用reduce)
%         H_to_group_5 = H_mul_user*conj(w_near_reshape); % K*Q,与H_to_group_1相同
%         H_to_group_5 = H_to_group_5.'; % Q*K
%         [Hr3_5,F3_5,setf3_5] = NF_group_5(H_to_group_5,K,H_to_group_5,N);
%         rf_num3_5 = N;  

        % NF_group_6(参考Ruicheng_Jiao的select函数,与NF_group_4类似,考虑了原select.m函数中使用的max_K)






        %% Near_field_NOMA 

        %% Near_field_NOMA_1 
        [SE_1,EE_1,ite_1,power1_1] = NF_NOMA(Hr3_1,F3_1,setf3_1,K,rf_num3_1,1/SNR,P,Imax);
        temp3_1 = temp3_1+SE_1;
%         temp33_1 = temp33_1+EE_1;
        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
%         index11 = 1 % 指示代码运行进程

        %% Near_field_NOMA_2  


        %% Near_field_NOMA_3  

        %% Near_field_NOMA_4
        [SE_4,EE_4,ite_4,power1_4] = NF_NOMA(Hr3_4,F3_4,setf3_4,K,rf_num3_4,1/SNR,P,Imax);
        temp3_4 = temp3_4+SE_4;
%         temp33_4 = temp33_4+EE_4;
        i_snr % 指示代码运行进程
        iter % 指示代码运行进程
%         index44 = 4 % 指示代码运行进程

        %% Near_field_NOMA_5
%         [SE_5,EE_5,ite_5,power1_5] = NF_NOMA(Hr3_5,F3_5,setf3_5,K,rf_num3_5,1/SNR,P,Imax);
%         temp3_5 = temp3_5+SE_5;
% %         temp33_5 = temp33_5+EE_5;
        %% Near_field_NOMA_6

        %% Near_field_OMA 

                                    
    end
    
    C_digital(i_snr) = temp/N_iter;
    capacity_near_near_digital(i_snr)= mean(sum_rate_near_near_digital);
    capacity_near_far_est(i_snr)= mean(sum_rate_near_far_est);
    capacity_near_near_est(i_snr)= mean(sum_rate_near_near_est);
    C3_1(i_snr) = temp3_1(end)/N_iter; % NOMA_1   
    C3_4(i_snr) = temp3_4(end)/N_iter; % NOMA_4 
        
end



%% 画图 plot
figure;
hold on 
plot(K_list,C_digital,'g-*','Linewidth',1.5); % 绿色
plot(K_list,capacity_near_near_digital,'r-*','Linewidth',1.5); % 红色
plot(K_list,capacity_near_far_est,'b-*','Linewidth',1.5); % 蓝色
plot(K_list,capacity_near_near_est,'y-*','Linewidth',1.5);% 黄色
plot(K_list,C3_1,'m-d','Linewidth',1.5); % 品红色 菱形
plot(K_list,C3_4,'m-s','Linewidth',1.5); % 品红色 方形
legend('Fully digital system-1', 'Fully digital system-2', 'SDMA(ZF)', 'LDMA(ZF)', 'Near-field NOMA-1','Near-field NOMA-4');
% legend('Fully digital system-1', 'Fully digital system-2', 'SDMA(ZF)', 'LDMA(ZF)', 'Near-field NOMA','Near-field NOMA-1','Near-field NOMA-2','Near-field NOMA-3','Near-field NOMA-4','Near-field NOMA-5','Near-field NOMA-6');
xlabel('K');
ylabel('Spectral efficiency');
grid on;
box on;