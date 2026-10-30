clear all;
clc;
close all;

Nt = 200;
Nr = 1;
fc = 30e9;
c = 3e8;
lambda_c = c/fc;
d = lambda_c / 2;
eps = 1e-3;
D_array = Nt*d;
R_frau = 2*D_array^2/lambda_c;

r_start_list = 5:1:100;
num_r = length(r_start_list);

start_point = 2^3;
end_point = 2^10;
sepate_point = 2^2;
Nt_list = start_point:sepate_point:end_point;
num_nt = length(Nt_list);

relation = zeros(num_r, num_nt);
relation2 = zeros(num_r, num_nt);
for i_r = 1:1
    r1_start = r_start_list(i_r);
    delta = 10;
    r1 = r1_start;
    r2 = r1_start+delta;

    theta = 30/180*pi;
    for i_iter = 1:num_nt
        Nt_iter = Nt_list(i_iter);
        h1 = near_field_manifold(Nt_iter, d, fc, r1, theta);
        h2 = near_field_manifold(Nt_iter, d, fc, r2, theta);
        h3 = near_field_approximation(Nt_iter, d, fc, r1, theta);
        h4 = near_field_approximation(Nt_iter, d, fc, r2, theta);
        
        relation(i_r, i_iter) = abs(h1*h2');
        relation2(i_r, i_iter) = abs(h3*h4');
    end
end
x_para = d^2*cos(theta)^2/lambda_c*(1/r_start_list(1)-1/(delta+r_start_list(1)));
beta = sqrt(x_para/2)*Nt_list;

num_point = 100000;
[f_sin, f_cos] = fresnel(beta, num_point);
relation_est = abs(f_sin+1i*f_cos)./beta;

figure;
hold on;
plot(Nt_list, relation2(1,:),'b-','Linewidth',1.2);
plot(Nt_list, relation_est,'g--','Linewidth',1.2);
legend('Calculated with near-field channel',...
    'Approximation with (14)');
xlabel('Number of antennas');
ylabel('Correlation');
grid on;
box on;
axis([0 Nt_list(end) 0 1]);