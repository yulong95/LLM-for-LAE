%% near_field GPR_beamtraining ceshi_Achievable rate vs distance 
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


% alpha_index = linspace(1,100,100);
% alpha_length = length(alpha_index);

sample = 500;
% sample = 300;


% SNR_dB = 10:2:30;
SNR_dB = 20;
SNR_linear = 10.^(SNR_dB/10.);

Distance_point = 5:5:30;



%% generate the far-field codebook 
disp('generate the far-field codebook')
s = 1;
D = s*N; 
row = (-(N - 1)/2:(N - 1)/2)' ;
col = -1 + 2/D : 2/D : 1 ;
theta_fn = asin(col);
DFT  =  exp( 1j*  pi * row * col ) / sqrt(N);
disp('the far-field codebook has been generated')
%% generate the near-field codebook
disp('generate the near-field codebook')

%
rho = 3;
rho_max = 64;
%

eta = 1.2;
rho_min = 4;
[w_near, sin_theta_list, r_list] = QuaCode_3(N, d, lambda_c, eta, rho_min);
for i = 1:length(r_list)
   r_list_2(i) = 1/r_list(i); % 1/r
end
num_theta = length(sin_theta_list);
num_r = length(r_list);
theta_delta = sin_theta_list(2) - sin_theta_list(1);
r_2_delta = r_list_2(2) - r_list_2(1);
location_delta = sqrt(theta_delta^2+r_2_delta^2);

disp('the near-field codebook has been generated')

%% generate the near-field hierarchical codebook
disp('generate the near-field hierarchical codebook')

P1 = [1,-1+2/D,40,2];
sampling_interval_1 = [2/D,2.3];
sampling_interval_2 = sampling_interval_1*A;
[w_hierarchical_1, theta_record_list_hierarchical_1,r_record_list_hierarchical_1] = QuaCode_hierarchical(N, d, lambda_c, P1,sampling_interval_2);
w_hierarchical_1 = w_hierarchical_1';
disp('the near-field hieerarchical codebook has been generated')

%%
rate_far = zeros(sample,length(Distance_point));
rate_near = zeros(sample,length(Distance_point));
rate_opt = zeros(sample,length(Distance_point));
rate_far_and_near = zeros(sample,length(Distance_point));
rate_near_hierarchical = zeros(sample,length(Distance_point));
rate_near_GPR = zeros(sample,length(Distance_point));
rate_near_GPR_ei = zeros(sample,length(Distance_point));
rate_near_GPR_gpucb = zeros(sample,length(Distance_point));


%% generate the Kernal
   disp('generate the Kernal')

   S = num_theta*num_r;
   location_index = zeros(S,2);
   %
   x = zeros(S,1);
   for xx = 1:S
       x(xx) = xx;
   end
   %
   index = 1;
   for i = 1:num_theta
       for j = 1:num_r
           location_index(index,1) = sin_theta_list(i);
           location_index(index,2) = r_list_2(j);
           index = index + 1;
       end
   end

   Kernal_exp = zeros(S,S);
   for aa = 1:S                                                                                                           
       for bb = 1:S
           Kernal_exp(aa,bb) = exp(-norm(location_index(aa,:)-location_index(bb,:))^2/(location_delta*1*0.5));
           % Kernal_exp(aa,bb) = exp(-norm(location_index(aa,:)-location_index(bb,:))^2/1);
       end
   end

   disp('the Kernal has been generated')


%% main_training

for t = 1:sample
    t  

  for s = 1:length(Distance_point)
    SNR = SNR_linear;
    distance = Distance_point(s);
    % generate the channel vector
    % [H, hc, r, theta, G] = near_field_channel(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 1);
    [H, hc, r, theta, G] = near_field_channel_distance(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 1,distance);
    H = channel_norm(H);
    k = 1;
    Hsf =  reshape(H(k, :, :), [N, M]);    
    Z = Hsf;
   % generate the Gain_vector
    Gain_vector = zeros(S,1);
    cnt = 1;
    for i = 1:num_theta
       for j = 1:num_r
           Gain_vector(cnt) = w_near(:,i,j)' * Z;
           cnt = cnt + 1;
       end
    end

    %% far-field beam training 
    array_gain_far = 0;
    for i =1:length(DFT)
        array_gain_far=max(array_gain_far,abs(conj(DFT(i,:))*Z)^2);
    end
    rate_far(t,s) = log2(1 + SNR * array_gain_far);
    
    %% near-field beam training
    array_gain_near = 0;
    cnt_beamtraining_near = 1;
    for i = 1:num_theta
       for j = 1:num_r
           if array_gain_near <= abs(w_near(:,i,j)' * Z)^2
              array_gain_near = abs(w_near(:,i,j)' * Z)^2;
              i_max = cnt_beamtraining_near;
           end
           cnt_beamtraining_near = cnt_beamtraining_near + 1;
       end
    end
    rate_near(t,s) = log2(1 + SNR * array_gain_near);

    %% perfect CSI based beamforming
    wc_opt = exp(1j*angle(Z'))/sqrt(N);
    array_gain = abs(wc_opt*Z)^2;
    rate_opt(t,s) = log2(1 + SNR * array_gain);
 
    % %% GPR_based_near_field beam training_ei
    % max_GPR_iter = 100;
    % % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_nearfield_ei_ceshi(Kernal_exp,SNR,1,S,max_GPR_iter,Gain_vector);
    % % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_nearfield_gpucb_ceshi(Kernal_exp,SNR,1,S,max_GPR_iter,Gain_vector);
    % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_nearfield_gpucb_ceshi_2(Kernal_exp,SNR,1,S,max_GPR_iter,Gain_vector);
    % mu_baseline = 0;
    % for i = 1:S
    %    if mu_baseline<=abs(mu_3(i))
    %       mu_baseline=abs(mu_3(i));
    %       GPR_index = i;
    %    end
    % end
    % 
    % %% fine search begin
    % Gain_baseline = 0;
    % if (GPR_index >= 20) && (GPR_index <= (S-20))
    % 
    % for i = (GPR_index-19):(GPR_index+20)
    %     if Gain_baseline <= abs(Gain_vector(i))^2
    %        Gain_baseline = abs(Gain_vector(i))^2;
    %        GPR_index = i;
    %    end
    % end
    % 
    % elseif (GPR_index >= 1) && (GPR_index <= 19)
    % 
    % for i = 1:20
    %     if Gain_baseline <= abs(Gain_vector(i))^2
    %        Gain_baseline = abs(Gain_vector(i))^2;
    %        GPR_index = i;
    %    end
    % end
    % 
    % elseif (GPR_index <= S) && (GPR_index >= (S-19))
    % 
    % for i = (S-19):S
    %    if Gain_baseline <= abs(Gain_vector(i))^2
    %        Gain_baseline = abs(Gain_vector(i))^2;
    %        GPR_index = i;
    %    end
    % end
    % 
    % 
    % end
    % % fine search end
    % 
    % % fine search 2 begin
    % 
    % for i = 1:max_GPR_iter
    %    if Gain_baseline <= abs(h_o_3(i))^2
    %       Gain_baseline = abs(h_o_3(i))^2;
    %       GPR_index = index_A_3(i);
    %    end
    % end
    % 
    % % fine search 2 end
    % %%
    % 
    % 
    % array_gain_GPR_near_ei = (abs(Gain_vector(GPR_index,1)))^2;
    % rate_near_GPR_ei(t,s) = log2(1 + SNR * array_gain_GPR_near_ei);
    % 
    % 
    % %% GPR_based_near_field beam training_gpucb
    % % max_GPR_iter = 100;
    % % [mu_3,cor_3,index_A_3,h_o_3,kmax_3] = GPR_beamtraining_nearfield_gpucb(Kernal_exp,SNR,1,S,max_GPR_iter,Gain_vector);
    % % mu_baseline = 0;
    % % for i = 1:S
    % %    if mu_baseline<=abs(mu_3(i))
    % %       mu_baseline=abs(mu_3(i));
    % %       GPR_index = i;
    % %    end
    % % end
    % 
    % % %% fine search begin
    % % Gain_baseline = 0;
    % % if (GPR_index >= 10) && (GPR_index <= (S-10))
    % % 
    % % for i = (GPR_index-9):(GPR_index+10)
    % %     if Gain_baseline <= abs(Gain_vector(i))^2
    % %        Gain_baseline = abs(Gain_vector(i))^2;
    % %        GPR_index = i;
    % %    end
    % % end
    % % 
    % % elseif (GPR_index >= 1) && (GPR_index <= 9)
    % % 
    % % for i = 1:10
    % %     if Gain_baseline <= abs(Gain_vector(i))^2
    % %        Gain_baseline = abs(Gain_vector(i))^2;
    % %        GPR_index = i;
    % %    end
    % % end
    % % 
    % % elseif (GPR_index <= S) && (GPR_index >= (S-9))
    % % 
    % % for i = (S-9):S
    % %    if Gain_baseline <= abs(Gain_vector(i))*H)^2
    % %        Gain_baseline = abs(Gain_vector(i))*H)^2;
    % %        GPR_index = i;
    % %    end
    % % end
    % % 
    % % 
    % % end
    % % % fine search end
    % % 
    % % % fine search 2 begin
    % % 
    % % for i = 1:max_GPR_iter
    % %    if Gain_baseline <= abs(h_o_3(i))^2
    % %       Gain_baseline = abs(h_o_3(i))^2;
    % %       GPR_index = index_A_3(i);
    % %    end
    % % end
    % % 
    % % % fine search 2 end
    % %%
    % 
    % 
    % array_gain_GPR_near_gpucb = (abs(Gain_vector(GPR_index,1)))^2;
    % rate_near_GPR_gpucb(t,s) = log2(1 + SNR * array_gain_GPR_near_gpucb);





    % %% Far_and_near_field beam training
    % % 1_far
    % array_gain_far_and_near_1 = 0;
    % for i =1:size(DFT,2)
    %     if array_gain_far_and_near_1<= abs(Z'*DFT(:,i))^2
    %        max_index_theta = theta_fn(i);
    %        array_gain_far_and_near_1 = abs(Z'*DFT(:,i))^2;
    %     end
    % end
    % select_max_theta(t,s) = max_index_theta;
    % % 2_near
    % w_far_and_near = QuaCode_fn(N, s, d, lambda_c, eta, rho,rho_max,max_index_theta);
    % w_far_and_near = w_far_and_near';
    % array_gain_far_and_near_2 = 0;
    % for i =1:size(w_far_and_near,1)
    %   if array_gain_far_and_near_2<=abs(w_far_and_near(i,:)*Z)^2
    %      array_gain_far_and_near_2=abs(w_far_and_near(i,:)*Z)^2;
    %   end
    % end
    % rate_far_and_near(t,s) = log2(1 + SNR * array_gain_far_and_near_2); 
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







%% plot
figure;
hold on
plot(Distance_point,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(Distance_point,mean(rate_near),'b^-','Linewidth',1.6)
plot(Distance_point,mean(rate_opt),'k--','Linewidth', 1.6)
plot(Distance_point,mean(rate_far_and_near),'cd-','Linewidth',1.6)
plot(Distance_point,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
plot(Distance_point,mean(rate_near_GPR_ei),'gp-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_gpucb),'yp-','Linewidth', 1.6)
legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)')
% legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)','Near-field Bayesian regression-based beam training(gpucb)')
xlabel('Distance (m)');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;









