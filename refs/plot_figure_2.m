%% plot
figure;
hold on
plot(alpha,Capacity,'ms-', 'Linewidth', 1.6)
plot(alpha,Proposedscheme,'b^-','Linewidth',1.6)
plot(alpha,CNN,'cd-','Linewidth',1.6)
plot(alpha,NFNOMA,'rd-','Linewidth', 1.6)
plot(alpha,NFLDMA,'gp-','Linewidth', 1.6)
plot(alpha,FFSDMA,'yp-','Linewidth', 1.6)
legend('Capacity','Proposed scheme','CNN','Near-field NOMA','Near-field LDMA','Far-field SDMA')
xlabel('$\alpha_{\mathrm{N}}$', 'Interpreter', 'latex');
ylabel('Spectrum Efficiency (bps/Hz)');
grid on;
box on;


