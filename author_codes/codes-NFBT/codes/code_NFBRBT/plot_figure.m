%% plot
figure;
hold on
plot(SNR_dB,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(SNR_dB,mean(rate_near),'b^-','Linewidth',1.6)
% plot(SNR_dB,mean(rate_opt),'k--','Linewidth', 1.6)
plot(SNR_dB,mean(rate_far_and_near),'cd-','Linewidth',1.6)
plot(SNR_dB,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
plot(SNR_dB,mean(rate_near_GPR_ei),'gp-','Linewidth', 1.6)
% plot(SNR_dB,mean(rate_near_GPR_gpucb),'yp-','Linewidth', 1.6)
legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Two-phase near-field beam training','Near-field hierarchical beam training','Proposed Bayesian regression-based beam training')
% legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)','Near-field Bayesian regression-based beam training(gpucb)')
xlabel('SNR (dB)');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;