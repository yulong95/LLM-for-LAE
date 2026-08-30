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

%% generate the near-field codebook 
disp('generate the near-field codebook')
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



%% generate the Kernal
   disp('generate the Kernal')
   S = num_theta*num_r;
   location_index = zeros(S,2);
   index = 1;
   
   for i = 1:num_theta
       for j = 1:num_r
           location_index(index,1) = sin_theta_list(i);
           location_index(index,2) = r_list_2(j);
           index = index + 1;
       end
   end

   % for i = 1:num_r
   %     for j = 1:num_theta
   %         location_index(index,1) = r_list_2(i);
   %         location_index(index,2) = sin_theta_list(j);
   %         index = index + 1;
   %     end
   % end


   col_far = -1 + 2/S : 2/S : 1 ;
   Kernal_exp = zeros(S,S);
   Kernal_J0 = zeros(S,S);
   Kernal_exp_far = zeros(S,S);
   for aa = 1:S                                                                                                           
       for bb = 1:S
           Kernal_exp(aa,bb) = exp(-norm(location_index(aa,:)-location_index(bb,:))^2/(location_delta*1));
           Kernal_exp_far(aa,bb) = exp(-norm(col_far(bb)-col_far(aa))^2/(2/S));
           Kernal_J0(aa,bb) = besselj(0,1*norm(location_index(aa,:)-location_index(bb,:))/(location_delta));
           % Kernal_exp(aa,bb) = exp(-norm(location_index(aa,:)-location_index(bb,:))^2/1);
       end
   end

%    Kernal_SV_mean = zeros(S,S);
%    Gain_vector_sum = zeros(S,1);
%    Rep = 10000;
%    for rp = 1:Rep
% 
%    [H, hc, r, theta, G] = near_field_channel(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 1);
%     H = channel_norm(H);
%     k = 1;
%     Hsf =  reshape(H(k, :, :), [N, M]);    
%     Z = Hsf;
%    % generate the Gain_vector
%     Gain_vector = zeros(S,1);
%     cnt = 1;
%     for i = 1:num_theta
%        for j = 1:num_r
%            Gain_vector(cnt) = w_near(:,i,j)' * Z;
%            cnt = cnt + 1;
%        end
%     end
% 
%        Gain_vector_sum = Gain_vector_sum +Gain_vector;
%    end 
%    Gain_vector_mean = Gain_vector_sum/Rep;
% 
%    for rp = 1:Rep
% 
% [H, hc, r, theta, G] = near_field_channel(N, K, L, d, fc, fs, M, Rmin, Rmax,sector, 1);
%     H = channel_norm(H);
%     k = 1;
%     Hsf =  reshape(H(k, :, :), [N, M]);    
%     Z = Hsf;
%    % generate the Gain_vector
%     Gain_vector = zeros(S,1);
%     cnt = 1;
%     for i = 1:num_theta
%        for j = 1:num_r
%            Gain_vector(cnt) = w_near(:,i,j)' * Z;
%            cnt = cnt + 1;
%        end
%     end
% 
%        Kernal_SV_mean = Kernal_SV_mean + (Gain_vector-Gain_vector_mean)*(Gain_vector-Gain_vector_mean)';
%    end
%    Kernal_SV_mean = Kernal_SV_mean/Rep;


   disp('the Kernal has been generated')



%% 计算矩阵A的元素幅值
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

    amplitudeMatrix = abs(Kernal_J0);
    % 使用imagesc函数画图
    figure; % 创建一个新的图形窗口
    imagesc(amplitudeMatrix); % 根据矩阵的幅值绘制图像
    colorbar; % 显示颜色条，以便于了解不同颜色对应的幅值大小
    title('Matrix Element Amplitude Visualization');
    axis equal tight; % 调整坐标轴的比例并紧密排布
    xlabel('Column Index');
    ylabel('Row Index');



