figure;
hold on; box on; grid on;
plot([1:N],real(h),'-b','LineWidth',1.5);
xlim([1,N])
scatter(index_A,real(h_o),50,'ob','LineWidth',1.5);
plot([1:N],real(h_hat_GPR),'k:','LineWidth',1.5);
shadedErrorBar([1:N],real(h_hat_GPR),beta*cor,'lineProps','b')
xlabel('Port index','Interpreter','latex','FontSize',18);
ylabel('Real part of ${\bf h}$','Interpreter','latex','FontSize',18);
legend('Truth','Sample','Mean','Location','southwest','FontSize',18,'Interpreter','latex');
axis square;
set(gca,'Fontname','Times','Fontsize',18);


figure;
hold on; box on; grid on;
plot([1:N],imag(h),'-r','LineWidth',1.5);
scatter(index_A,imag(h_o),50,'or','LineWidth',1.5);
plot([1:N],imag(h_hat_GPR),'k:','LineWidth',1.5);
shadedErrorBar([1:N],imag(h_hat_GPR),beta*cor,'lineProps','r')
xlim([1,N])
xlabel('Port index','Interpreter','latex','FontSize',18);
ylabel('Imaginary part of ${\bf h}$','Interpreter','latex','FontSize',18);
legend('Truth','Sample','Mean','Location','northwest','FontSize',18,'Interpreter','latex');
axis square;
set(gca,'Fontname','Times','Fontsize',18);