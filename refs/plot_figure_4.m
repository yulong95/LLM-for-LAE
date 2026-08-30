%% plot
figure;
hold on
plot(SNR_dB,Capacity,'ms-', 'Linewidth', 1.6)
plot(SNR_dB,Proposedscheme,'b^-','Linewidth',1.6)
plot(SNR_dB,CNN,'cd-','Linewidth',1.6)
plot(SNR_dB,NFNOMA,'rd-','Linewidth', 1.6)
plot(SNR_dB,NFLDMA,'gp-','Linewidth', 1.6)
plot(SNR_dB,FFSDMA,'yp-','Linewidth', 1.6)
legend('Capacity','Proposed scheme','CNN','Near-field NOMA','Near-field LDMA','Far-field SDMA')
xlabel('BS transmit power (dBW)');
ylabel('Spectrum Efficiency (bps/Hz)');
grid on;
box on;