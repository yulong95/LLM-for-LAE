%% near_field GPR_beamtraining
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

% sample = 500;
sample = 50;


SNR_dB = 10:2:30;
SNR_linear = 10.^(SNR_dB/10.);


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

P1 = [1,-1+2/D,64,2];
sampling_interval_1 = [2/D,1];
sampling_interval_2 = sampling_interval_1*A;
[w_hierarchical_1, theta_record_list_hierarchical_1,r_record_list_hierarchical_1] = QuaCode_hierarchical(N, d, lambda_c, P1,sampling_interval_2);
w_hierarchical_1 = w_hierarchical_1';
disp('the near-field hieerarchical codebook has been generated')

%%
rate_far = zeros(sample,length(SNR_dB));
rate_near = zeros(sample,length(SNR_dB));
rate_opt = zeros(sample,length(SNR_dB));
rate_far_and_near = zeros(sample,length(SNR_dB));
rate_near_hierarchical = zeros(sample,length(SNR_dB));
rate_near_GPR = zeros(sample,length(SNR_dB));
rate_near_GPR_ei = zeros(sample,length(SNR_dB));
rate_near_GPR_gpucb = zeros(sample,length(SNR_dB));


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

 %%

    % 计算矩阵A的元素幅值
    amplitudeMatrix = abs(Kernal_exp);
    % 使用imagesc函数画图
    figure; % 创建一个新的图形窗口
    imagesc(amplitudeMatrix); % 根据矩阵的幅值绘制图像
    colorbar; % 显示颜色条，以便于了解不同颜色对应的幅值大小
    title('Matrix Element Amplitude Visualization');
    axis equal tight; % 调整坐标轴的比例并紧密排布
    xlabel('Column Index');
    ylabel('Row Index');

  %%

    % 计算矩阵A的元素幅值
    amplitudeMatrix = abs(Kernal_exp);
    % 使用imagesc函数画图
    figure; % 创建一个新的图形窗口
    imagesc(amplitudeMatrix); % 根据矩阵的幅值绘制图像
    colorbar; % 显示颜色条，以便于了解不同颜色对应的幅值大小
    % title('Matrix Element Amplitude Visualization');
    axis equal tight; % 调整坐标轴的比例并紧密排布
    xlabel('Column Index');
    ylabel('Row Index');



