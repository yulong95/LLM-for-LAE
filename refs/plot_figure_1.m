%% plot
figure;
hold on
plot(K,Capacity,'ms-', 'Linewidth', 1.6)
plot(K,Proposedscheme,'b^-','Linewidth',1.6)
plot(K,CNN,'cd-','Linewidth',1.6)
plot(K,NFNOMA,'rd-','Linewidth', 1.6)
plot(K,NFLDMA,'gp-','Linewidth', 1.6)
plot(K,FFSDMA,'yp-','Linewidth', 1.6)
legend('Capacity','Proposed scheme','CNN','Near-field NOMA','Near-field LDMA','Far-field SDMA')
xlabel('The number of users');
ylabel('Spectrum Efficiency (bps/Hz)');
grid on;
box on;