%% plot
figure;
hold on
plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(SNR_dB,mean(rate_near),'b^-','Linewidth',1.6)
plot(SNR_dB,mean(rate_opt),'k--','Linewidth', 1.6)
plot(SNR_dB,mean(rate_far_and_near),'cd-','Linewidth',1.6)
plot(SNR_dB,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
plot(SNR_dB,mean(rate_near_GPR_ei),'gp-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_gpucb),'yp-','Linewidth', 1.6)
legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)')
% legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)','Near-field Bayesian regression-based beam training(gpucb)')
xlabel('SNR (dB)');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;




%%
figure;
hold on
plot(x,abs(mu_3),'ms--', 'Linewidth', 1.6)
grid on;
box on;
%%
figure;
hold on
plot(x,abs(Gain_vector),'cs-', 'Linewidth', 1.6)
grid on;
box on;
%%
figure;
hold on
plot(x,abs(mu_3),'ms--', 'Linewidth', 1.6)
grid on;
box on;

figure;
hold on
plot(x,abs(Gain_vector),'cs-', 'Linewidth', 1.6)
grid on;
box on;





%% plot_kernal
Gain_vector_final = Gain_vector/max(abs(Gain_vector));
mu_3_final = mu_3/max(abs(mu_3));
figure;
hold on
plot(x,abs(mu_3_final),'r.-', 'Linewidth', 0.8)
plot(x,abs(Gain_vector_final),'c.-', 'Linewidth', 0.8)
legend('Truth','Reconstructed')
xlabel('Location index');
ylabel('Normalized beamforming gain');
grid on;
box on;





















