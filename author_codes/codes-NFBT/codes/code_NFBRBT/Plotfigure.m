
figure;
subplot(2,3,1);
hold on; box on; grid on;
subtitle('Real part');
plot([1:N],real(h),'-b');
% plot(x,real(h_hat_GPR) + beta*cor,':k');
% plot(x,real(h_hat_GPR) - beta*cor,':k');
xlim([1,N])
scatter(index_A,real(h_o),'ob');
shadedErrorBar([1:N],real(h_hat_GPR),beta*cor,'lineProps','b')
xlabel('x');
ylabel('y');
legend('Truth','Sample','Location','southwest');
axis square;

subplot(2,3,4);
hold on; box on; grid on;
subtitle('Real part');
plot([1:N],real(h),'-b');
plot([1:N],real(h_hat_GPR),'kx');
xlabel('x');
ylabel('y');
xlim([1,N])
legend('Truth','Reconstruct','Location','southwest');
axis square;

subplot(2,3,2);
hold on; box on; grid on;
subtitle('Imag part');
plot([1:N],imag(h),'-r');
% plot(x,imag(h_hat_GPR) + beta*cor,':k');
% plot(x,imag(h_hat_GPR) - beta*cor,':k');
scatter(index_A,imag(h_o),'or');
shadedErrorBar([1:N],imag(h_hat_GPR),beta*cor,'lineProps','r')
xlim([1,N])
xlabel('x');
ylabel('y');
legend('Truth','Sample','Location','southwest');
axis square;

subplot(2,3,5);
hold on; box on; grid on;
subtitle('Imag part');
plot([1:N],imag(h),'-r');
plot([1:N],imag(h_hat_GPR),'kx');
xlim([1,N])
xlabel('x');
ylabel('y');
legend('Truth','Reconstruct','Location','southwest');
axis square;

subplot(2,3,3);
hold on; box on; grid on;
subtitle('Amplitude');
plot([1:N],abs(h),'-r');
plot([1:N],abs(h_hat_GPR),'kx');
xlim([1,N])
xlabel('x');
ylabel('amplitude');
legend('Truth','Reconstruct','Location','southwest');
axis square;

subplot(2,3,6);
hold on; box on; grid on;
subtitle('Angle');
plot([1:N],angle(h),'-r');
plot([1:N],angle(h_hat_GPR),'kx');
xlim([1,N])
xlabel('x');
ylabel('angle');
legend('Truth','Reconstruct','Location','southwest');
axis square;