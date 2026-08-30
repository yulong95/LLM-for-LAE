function [Hr,F,setf] = NF_group_5(H_est,K,H,rf_num)
% 不采用reduce
% reduce_RF.m和select.m结合,即不采用reduce

% H_to_group_5 = H_mul_user*conj(w_near_reshape); % K*Q,与H_to_group_1相同
% H_to_group_5 = H_to_group_5.'; % Q*K
%[Hr3_5,F3_5,setf3_5] = NF_group_5(H_to_group_5,K,H_to_group_5,N);

H_abs = abs(H_est);
set = [];

% 每个用户最强beam
for k = 1:K
    [~,order] = sort(H_abs(:,k),'descend');
    set = [set;order(1)];
end 

[x1,ind] = sort(set);
[x2,m,~] = unique(x1,'last');  % x2保留的beam
rf_x2 = length(x2);    

%%
integer_index = (1:K); % 1到K的正整数数列
integer_index = integer_index.';
x3 = zeros(rf_num,1);
mm = zeros(rf_num,1);

if rf_x2 >= rf_num
   for i = 1:rf_num
       x3(i) = x2(i);
       mm(i) = m(i);
   end
else
   for ii = 1:rf_num
       x3(ii) = x1(ii);
       mm(ii) = integer_index(ii);
   end
end




% 假设用户数设置为32，射频链数设置为16
% 则x1的个数有32个，32个中有重复的，而且已经按升序排好序
% x2在x1的基础上保留唯一，不一定多少个，但是数量一定比32少，大概率大于16
% 因此该部分的思路就是让x3在x2的基础上就是固定的16，没考虑如果x2的数量比16小的情况，后续有待改善
% 运行有bug,因此需要改善
%%
Hr = H(x3,:);
Hr_e = H_est(x3,:);



Hr_e0 = Hr_e;
Ht_abs = abs(Hr_e);
j  = 0;
mov = []; % 弱用户集合
setf = zeros(rf_num,K); % 每个波束的用户集合
for i = 1:rf_num
    snum = mm(i)-j;  % 第i个beam的用户数
    Ht = Ht_abs(:,ind(j+1:mm(i)));
    Ht = sum(Ht.^2,1);
    [~,order] = sort(Ht,'descend');
    setf(i,1:snum) = ind(j+order);  % 第i个beam的用户，信道增益由强到弱
    mov = [mov; ind(j+order(2:end))];
    j = mm(i);   
end

sett = 1:K;
setr = setdiff(sett',mov); % 强用户集合
    
Hr_e0(:,mov) = [];  % 每个beam的最强用户信道作为代表信道
Ft = Hr_e0*inv(Hr_e0'*Hr_e0);  % ZF precoding
F = zeros(rf_num,K);
F(:,setr) = Ft;
j  = 0;
for i = 1:rf_num
    Ht = Ht_abs(:,ind(j+1:mm(i)));
    Ht = sum(Ht.^2,1); 
    [~,order] = sort(Ht,'descend');
    for k = 2:length(order)
        F(:,ind(j+order(k))) = F(:,ind(j+order(1)));
    end
    j = mm(i);
end