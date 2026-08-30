clear all; clc;

f_c = 2.4e9;        % Frequency

Lambda = 3e8/f_c;   % Wavelength

sigma2_dB = -20;

sigma2 = 10.^(sigma2_dB/10);   % Noise power

beta = 3;

P = 10;
N_ports = 4;

L_num = [2:4:38];

W = 10;

Length = length(L_num);

Rep = 2000;

NMSE_GPR = zeros(Length,Rep);
NMSE_GPR_exp = zeros(Length,Rep);
NMSE_GPR_J0 = zeros(Length,Rep);
NMSE_OMP = zeros(Length,Rep);
NMSE_ML = zeros(Length,Rep);
NMSE_LMMSE = zeros(Length,Rep);

for qq = 1:Length

    load(['Kernel_L/Kernal' num2str(L_num(qq)) '.mat']);

    parfor rp = 1:Rep
        % rng(rp);
        % h = U*S^(1/2)*(randn(N,1)+1j*randn(N,1))/sqrt(2);
        % h = CDL_channel(x, f_c, Lambda);
        h = SV_channel(x,Lambda,L);

        h = h/sqrt(mean(diag(Kernal_CDL_mean)));

        [h_hat_GPR,~,~,~] = GPR_multi(x,h,Kernal_SV_mean, sigma2, P*N_ports, 1);
        [h_hat_J0,~,~,~] = GPR_multi(x,h,Kernal_J0, sigma2, P*N_ports, 1);
        % [h_hat_exp,~,~,~] = GPR_multi(x,h,Kernal_exp, sigma2, P*N_ports, 1);
        [h_hat_ML, h_hat_OMP] = ML_OMP_estimator(h, P, N_ports, 2*L, sigma2, 20);
        [h_hat_LMMSE] = SeLMMSE(x,h,sigma2,P*N_ports);

        NMSE_GPR(qq,rp) = mag2db(norm(h_hat_GPR - h)/norm(h));
        NMSE_LMMSE(qq,rp) = mag2db(norm(h_hat_LMMSE - h)/norm(h));
        NMSE_GPR_J0(qq,rp) = mag2db(norm(h_hat_J0 - h)/norm(h));
        % NMSE_GPR_exp(qq,rp) = mag2db(norm(h_hat_exp - h)/norm(h));
        NMSE_OMP(qq,rp) = mag2db(norm(h_hat_OMP - h)/norm(h));
        NMSE_ML(qq,rp) = mag2db(norm(h_hat_ML - h)/norm(h));

        fprintf("Rep %d complete.\n", qq);
    end
end

NMSE_GPR = mean(NMSE_GPR,2);
NMSE_GPR_J0 = mean(NMSE_GPR_J0,2);
NMSE_GPR_exp = mean(NMSE_GPR_exp,2);
NMSE_OMP = mean(NMSE_OMP,2);
NMSE_ML = mean(NMSE_ML,2);
NMSE_LMMSE = mean(NMSE_LMMSE,2);

save('NMSE_vs_L_SV.mat','L_num','NMSE_OMP','NMSE_ML','NMSE_LMMSE','NMSE_GPR_exp','NMSE_GPR');


C = linspecer(5);

%% Plot the curve.
figure;
box on; grid on; hold on;
plot(L_num,NMSE_OMP,'-s','LineWidth',1.5,'Color',C(4,:));
plot(L_num,NMSE_ML,'-o','LineWidth',1.5,'Color',C(3,:));
plot(L_num,NMSE_LMMSE,'-p','LineWidth',1.5,'Color',C(5,:));
plot(L_num,NMSE_GPR_J0,'-r>','LineWidth',1.5,'Color',C(2,:));
plot(L_num,NMSE_GPR,'-d','LineWidth',1.5,'Color',C(1,:));
legend('FAS-OMP','FAS-ML','SeLMMSE','Proposed $\bar S$ (${\bf \Sigma}_{\rm bes}$)','Proposed $\bar S$ (${\bf \Sigma}_{\rm cov}$)','FontSize',12,'Interpreter','latex');
xlabel('Number of paths $L$','Interpreter','latex','FontSize',14);
ylabel('NMSE (dB)','Interpreter','latex','FontSize',14);

