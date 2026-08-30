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

SNR_dB = -10:2:10;
SNR_linear = 10.^(SNR_dB/10.);




% generate the far-field codebook 
disp('generate the far-field codebook')

s = 1;
D = s*N; 
Codebook_far = zeros(D,N);
col = -1 + 2/D : 2/D : 1 ;
theta_fn = asin(col);
for i = 1:D
    Codebook_far(i,:) = array_respones(theta_fn(i),N,d,lambda_c);
end
S = size(Codebook_far,1);

x = zeros(S,1);
for xx = 1:S
    x(xx) = xx;
end

disp('the far-field codebook has been generated')


rate_far = zeros(sample,length(SNR_dB));
rate_GPR = zeros(sample,length(SNR_dB));
rate_perfect = zeros(sample,length(SNR_dB));
%% generate the Kernal
   disp('generate the Kernal')
   Rep = 10000;
   Kernal_SV_mean = zeros(S,S);

   Kernal_exp = zeros(S,S);
   for aa = 1:S
    for bb = 1:S
        % Kernal_exp(aa,bb) = exp(-norm(theta_fn(bb)-theta_fn(aa))^2);
        % Kernal_exp(aa,bb) = exp(-norm(col(bb)-col(aa))^2/1);
        % Kernal_exp(aa,bb) = exp(-norm(col(bb)-col(aa))^2/0.01);
        Kernal_exp(aa,bb) = exp(-norm(col(bb)-col(aa))^2/(2/D));
    end
   end

   % Gain_vector_sum = zeros(S,1);

   % for rp = 1:Rep
   %     H = farfield_channel(N,K,L,lambda_c,d);
   %     Gain_vector = conj(Codebook_far)*H;
   %     Gain_vector_sum = Gain_vector_sum +Gain_vector;
   % end 
   % Gain_vector_mean = Gain_vector_sum/Rep;
   % 
   % for rp = 1:Rep
   %     H = farfield_channel(N,K,L,lambda_c,d);
   %     Gain_vector = conj(Codebook_far)*H;
   %     Kernal_SV_mean = Kernal_SV_mean + (Gain_vector-Gain_vector_mean)*(Gain_vector-Gain_vector_mean)';
   % end
   % Kernal_SV_mean = Kernal_SV_mean/Rep;
   disp('the Kernal has been generated')

   GPR_train_num_average = zeros(sample,length(SNR_dB));

%% training

for t = 1:sample
    t   
    H = farfield_channel(N,K,L,lambda_c,d);

   % generate the Gain_vector
    Gain_vector = conj(Codebook_far)*H;
    

 
   for s = 1:length(SNR_dB)
      s
    SNR = SNR_linear(s);
    %% far-field beam training 
    array_gain_far = 0;
    for i =1:length(Codebook_far)
         if array_gain_far<=abs(conj(Codebook_far(i,:))*H)^2
            i_max = i;
            array_gain_far=abs(conj(Codebook_far(i,:))*H)^2;
         end
    end
    rate_far(t,s) = log2(1 + SNR * array_gain_far);  
    %% GPR_based_far_field beam training
    max_GPR_iter = 50;
    % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_5(Kernal_SV_mean,SNR,1,S,max_GPR_iter,Codebook_far,Gain_vector);
    % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_6(Kernal_SV_mean,SNR,1,S,max_GPR_iter,Codebook_far,Gain_vector);
    % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_6(Kernal_exp,SNR,1,S,max_GPR_iter,Codebook_far,Gain_vector);
    [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_10_2(Kernal_exp,SNR,1,S,max_GPR_iter,Codebook_far,Gain_vector);
    % [mu_3,cor_3,index_A_3,h_o_3,kmax_3,GPR_index,GPR_train_number] = GPR_beamtraining_10(Kernal_exp,SNR,1,S,max_GPR_iter,Codebook_far,Gain_vector,i_max);
    % GPR_baseline = 0;
    % for bbb = 1:max_GPR_iter
    %     if GPR_baseline <= abs(h_o_3(bbb))
    %        GPT_baseline = abs(h_o_3(bbb));
    %        GPR_index = index_A_3(bbb);
    %     end
    % end
    [h_o_3_max_value,h_o_3_max_index] = max(h_o_3);
    GPR_index = index_A_3(h_o_3_max_index);
    array_gain_far_GPR= abs(conj(Codebook_far(GPR_index,:))*H)^2;
    rate_GPR(t,s) = log2(1 + SNR * array_gain_far_GPR);

    %% Perfect CSI beamforming
    wc_opt = exp(1j*angle(H'))/sqrt(N);
    array_gain_perfect = abs(wc_opt*H)^2;
    rate_perfect(t,s) = log2(1 + SNR * array_gain_perfect); 
   end
end



%% plot
figure;
hold on
plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(SNR_dB,mean(rate_GPR),'gp-','Linewidth', 1.6)
plot(SNR_dB,mean(rate_perfect),'rd-', 'Linewidth', 1.6)
legend('Far-field beam training ', 'GPR based far field beam training ','Perfect CSI beamforming')
xlabel('SNR (dB)');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;

% figure;
% hold on
% plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
% plot(SNR_dB,mean(rate_perfect),'rd-', 'Linewidth', 1.6)
% legend('Far-field beam training ','Perfect CSI beamforming')
% xlabel('SNR (dB)');
% ylabel('Achievable Rate (bis/s/Hz)');
% grid on;
% box on;




