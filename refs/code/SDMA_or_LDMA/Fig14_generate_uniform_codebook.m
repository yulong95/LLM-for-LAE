clear all;
clc;
close all;

c = 3e8;
Nt1 = 256;
Nt2 = 16;
fc = 30e9;
lambda = c/fc; % wavelegenth;
d = lambda/2;

D = sqrt((Nt1*d)^2+(Nt2*d)^2);
D_rayleigh = 2*D^2/lambda;
D_fresnel = D/2*(D/lambda)^(1/3);

%% calculate G(beta1) *  G(beta2) and derive r for each angle
x_range = [0, 50];
y_range = [-50, 50];
z_range = [-50, 50];
num_point = 21676;

x_scope = x_range(2)-x_range(1);
y_scope = y_range(2)-y_range(1);
z_scope = z_range(2)-z_range(1);
y_scale = y_scope/x_scope;
z_scale = z_scope/x_scope;

num_x = nthroot(num_point/y_scale/z_scale, 3);
x_list = linspace(x_range(1), x_range(2), num_x);
y_list = linspace(y_range(1), y_range(2), num_x*y_scale);
z_list = linspace(z_range(1), z_range(2), num_x*z_scale);
num_w = length(x_list)*length(y_list)*length(z_list);

w = Code3D_uniform(Nt1, Nt2, d, lambda, x_list, y_list, z_list);

correlation = abs(w'*w);
correlation(logical(eye(size(correlation))))=0;
[correlation_max, max_idx] = max(correlation);

save('Codebook3d_30G_256_16_uniform.mat');