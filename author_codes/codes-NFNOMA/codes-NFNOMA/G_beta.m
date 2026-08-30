clear all;
clc;
close all;
%%
beta_list = linspace(0.001,10,2000);
num_point = 1000;
G = zeros(1,length(beta_list));

%% 
for i = 1:length(beta_list)
    beta = beta_list(i);
    [f_sin,f_cos] = fresnel(beta,num_point);
    G(i) = abs((f_cos+1i*f_sin)/beta);
end

%%  plot
figure;
hold on 
plot(beta_list,G,'r-*','Linewidth',1.5); % 红色
xlabel('beta');
ylabel('G');
grid on;
box on;