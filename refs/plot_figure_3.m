%% plot
figure;
hold on
plot(Rmin,Capacity,'ms-', 'Linewidth', 1.6)
plot(Rmin,Proposedscheme,'b^-','Linewidth',1.6)
plot(Rmin,CNN,'cd-','Linewidth',1.6)
plot(Rmin,NFNOMA,'rd-','Linewidth', 1.6)
plot(Rmin,NFLDMA,'gp-','Linewidth', 1.6)
plot(Rmin,FFSDMA,'yp-','Linewidth', 1.6)
legend('Capacity','Proposed scheme','CNN','Near-field NOMA','Near-field LDMA','Far-field SDMA')
xlabel('$R_{\min}$ (bps/s/Hz)', 'Interpreter', 'latex');
ylabel('Spectrum Efficiency (bps/Hz)');
grid on;
box on;


