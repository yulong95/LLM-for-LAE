%% plot
figure;
hold on
plot(overhead_index,mean(rate_far),'ms-', 'Linewidth', 1.6)
plot(overhead_index,mean(rate_near),'b^-','Linewidth',1.6)
% plot(overhead_index,mean(rate_opt),'k--','Linewidth', 1.6)
plot(overhead_index,mean(rate_far_and_near),'cd-','Linewidth',1.6)
plot(overhead_index,mean(rate_near_hierarchical),'rd-','Linewidth', 1.6)
plot(overhead_index,rate_near_GPR_result(sample,:),'gp-','Linewidth', 1.6)
% plot(overhead_index,mean(rate_near_GPR_gpucb),'yp-','Linewidth', 1.6)
legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Two-phase near-field beam training','Near-field hierarchical beam training','Proposed Bayesian regression-based beam training')
% legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)','Near-field Bayesian regression-based beam training(gpucb)')
xlabel('Beam training overhead');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;
%% plot 1
overhead_index_1 = [1:5:100];
rate_far_final = mean(rate_far);
rate_far_final = rate_far_final(overhead_index_1);
rate_near_final = mean(rate_near);
rate_near_final = rate_near_final(overhead_index_1);
rate_far_and_near_final = mean(rate_far_and_near);
rate_far_and_near_final = rate_far_and_near_final(overhead_index_1);
rate_near_hierarchical_final = mean(rate_near_hierarchical);
rate_near_hierarchical_final = rate_near_hierarchical_final(overhead_index_1);
rate_near_GPR_result_final = rate_near_GPR_result(sample,:);
rate_near_GPR_result_final = rate_near_GPR_result_final(overhead_index_1);

figure;
hold on
plot(overhead_index_1,rate_far_final,'ms-', 'Linewidth', 1.6)
plot(overhead_index_1,rate_near_final,'b^-','Linewidth',1.6)
% plot(overhead_index,mean(rate_opt),'k--','Linewidth', 1.6)
plot(overhead_index_1,rate_far_and_near_final,'cd-','Linewidth',1.6)
plot(overhead_index_1,rate_near_hierarchical_final,'rd-','Linewidth', 1.6)
plot(overhead_index_1,rate_near_GPR_result_final,'gp-','Linewidth', 1.6)
% plot(overhead_index,mean(rate_near_GPR_gpucb),'yp-','Linewidth', 1.6)
legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Two-phase near-field beam training','Near-field hierarchical beam training','Proposed Bayesian regression-based beam training')
% legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)','Near-field Bayesian regression-based beam training(gpucb)')
xlabel('Beam training overhead');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;

%% plot 2
overhead_index_1 = [1:50:1275];
rate_far_final = mean(rate_far);
rate_far_final = rate_far_final(overhead_index_1);
rate_near_final = mean(rate_near);
rate_near_final = rate_near_final(overhead_index_1);
rate_far_and_near_final = mean(rate_far_and_near);
rate_far_and_near_final = rate_far_and_near_final(overhead_index_1);
rate_near_hierarchical_final = mean(rate_near_hierarchical);
rate_near_hierarchical_final = rate_near_hierarchical_final(overhead_index_1);
rate_near_GPR_result_final = rate_near_GPR_result(sample,:);
rate_near_GPR_result_final = rate_near_GPR_result_final(overhead_index_1);

figure;
hold on
plot(overhead_index_1,rate_far_final,'ms-', 'Linewidth', 1.6)
plot(overhead_index_1,rate_near_final,'b^-','Linewidth',1.6)
% plot(overhead_index,mean(rate_opt),'k--','Linewidth', 1.6)
plot(overhead_index_1,rate_far_and_near_final,'cd-','Linewidth',1.6)
plot(overhead_index_1,rate_near_hierarchical_final,'rd-','Linewidth', 1.6)
plot(overhead_index_1,rate_near_GPR_result_final,'gp-','Linewidth', 1.6)
% plot(overhead_index,mean(rate_near_GPR_gpucb),'yp-','Linewidth', 1.6)
legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Two-phase near-field beam training','Near-field hierarchical beam training','Proposed Bayesian regression-based beam training')
% legend('Far-field exhaustive beam training ','Near-field exhaustive beam training','Perfect CSI based beamforming','Two-phase near-field beam trainining','Near-field hierarchical beam training','Near-field Bayesian regression-based beam training(ei)','Near-field Bayesian regression-based beam training(gpucb)')
xlabel('Beam training overhead');
ylabel('Achievable Rate (bis/s/Hz)');
grid on;
box on;


