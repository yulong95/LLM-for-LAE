function [Hr,F,rf_num,setf] = NF_group_1(H_est,K,H)
% [Hr3_1,F3_1,rf_num3_1,setf3_1] = NF_group_1(H_to_group_1,K,H_to_group_1);

H_abs = abs(H_est);
set = [];

% 每个用户最强beam
for k = 1:K
    [~,order] = sort(H_abs(:,k),'descend');
    set = [set;order(1)];
end 

[x1,ind] = sort(set);
[x2,m,~] = unique(x1,'last');  % x2保留的beam
rf_num = length(x2);    % 射频数

Hr = H(x2,:);
Hr_e = H_est(x2,:);

Hr_e0 = Hr_e;
Ht_abs = abs(Hr_e);
j  = 0;
mov = []; % 弱用户集合
setf = zeros(rf_num,K); % 每个波束的用户集合
for i = 1:rf_num
    snum = m(i)-j;  % 第i个beam的用户数
    Ht = Ht_abs(:,ind(j+1:m(i)));
    Ht = sum(Ht.^2,1);
    [~,order] = sort(Ht,'descend');
    setf(i,1:snum) = ind(j+order);  % 第i个beam的用户，信道增益由强到弱
    mov = [mov; ind(j+order(2:end))];
    j = m(i);   
end

sett = 1:K;
setr = setdiff(sett',mov); % 强用户集合
    
Hr_e0(:,mov) = [];  % 每个beam的最强用户信道作为代表信道
Ft = Hr_e0*inv(Hr_e0'*Hr_e0);  % ZF precoding
F = zeros(rf_num,K);
F(:,setr) = Ft;
j  = 0;
for i = 1:rf_num
    Ht = Ht_abs(:,ind(j+1:m(i)));
    Ht = sum(Ht.^2,1); 
    [~,order] = sort(Ht,'descend');
    for k = 2:length(order)
        F(:,ind(j+order(k))) = F(:,ind(j+order(1)));
    end
    j = m(i);
end