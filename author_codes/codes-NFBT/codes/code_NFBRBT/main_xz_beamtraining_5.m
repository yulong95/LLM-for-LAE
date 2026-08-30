%% Try the performance of GPR_based far-field beamtraining (DFT codebook)(based on the version 4)
%% 
clc;
clear all
close all

N = 256; % the number of the antennas at the BS
K = 1;% the number of users
M = 1;% number of subcarriers
L = 3; % number of paths per user

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
disp('generate the far-field codebook')
s = 1;
D = s*N; 
row = (-(N - 1)/2:(N - 1)/2)' ;
col = -1 + 2/D : 2/D : 1 ;
theta_fn = asin(col);
DFT  =  exp( 1j*  pi * row * col ) / sqrt(N);
S = length(DFT);
disp('the far-field codebook has been generated')


rate_far = zeros(sample,length(SNR_dB));
rate_GPR = zeros(sample,length(SNR_dB));

%% generate the Kernal
    H = farfield_channel(N,K,L,lambda_c,d);

   % generate the Gain_vector
    Gain_vector = DFT*H;
 
   % generate the Kernal_2
   disp('generate the Kernal')
   Rep = 100;
   Kernal_SV_mean = zeros(S,S);
   for rp = 1:Rep
       Kernal_SV_mean = Kernal_SV_mean + Gain_vector*Gain_vector';
   end
   Kernal_SV_mean = Kernal_SV_mean/Rep;
   disp('the Kernal has been generated')

   GPR_train_num_average = zeros(sample,length(SNR_dB));

%% training

for t = 1:sample
    t   
    H = farfield_channel(N,K,L,lambda_c,d);

   % generate the Gain_vector
    Gain_vector = DFT*H;
 
   for s = 1:length(SNR_dB)
      s
    SNR = SNR_linear(s);
    %% far-field beam training 
    array_gain_far = 0;
    for i =1:length(DFT)
         if array_gain_far<=abs(DFT(i,:)*H)^2
            i_max = i;
            array_gain_far=abs(DFT(i,:)*H)^2;
         end
    end
    rate_far(t,s) = log2(1 + SNR * array_gain_far);  
    %% GPR_based_far_field beam training
    max_GPR_iter = 1000;
    [mu_3,cor_3,index_A_3,h_o_3,kmax_3,GPR_index_3,GPR_train_number,aa] = GPR_beamtraining_3(Kernal_SV_mean,SNR,1,S,max_GPR_iter,DFT,Gain_vector,i_max);
    GPR_train_num_average(t,s) = GPR_train_number;
    array_gain_far_GPR= abs(DFT(GPR_index_3,:)*H)^2;
    rate_GPR(t,s) = log2(1 + SNR * array_gain_far_GPR);
   end
end



%% plot
figure;
hold on
plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(SNR_dB,mean(rate_GPR),'gp-','Linewidth', 1.6)
legend('Far-field beam training ', 'GPR based far field beam training ')
xlabel('SNR (dB)');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;

