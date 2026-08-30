
figure;
hold on 
plot(delta_list,C3_1,'r-*','Linewidth',1.5); % 红色
plot(delta_list,C3_4,'g-*','Linewidth',1.5); % 绿色
legend('Near-field NOMA-1','Near-field NOMA-4');
xlabel('beta');
ylabel('SE');
grid on;
box on;


figure;
hold on 
plot(delta_list,G,'r-*','Linewidth',1.5); % 红色
xlabel('beta');
ylabel('G');
grid on;
box on;


figure;
hold on 
plot(G,C3_1,'r-*','Linewidth',1.5); % 红色
plot(G,C3_4,'g-*','Linewidth',1.5); % 绿色
legend('Near-field NOMA-1','Near-field NOMA-4');
xlabel('delta');
ylabel('SE');
grid on;
box on;






