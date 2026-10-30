function DP = D_precoder_NF(H_eq,AP,K,N_RF,setf)

He = H_eq(setf(:,1),:);
D_hat = He'*inv(He*He');
% D_hat = D_hat./repmat(sqrt(sum(abs(AP*D_hat).^2,1)),N_RF,1); % 这里大概率不需要做归一化，因为后面的NF_NOMA函数中做了归一化(如果最后值很小的话把这一行加上，不归一化)

DP = zeros(N_RF,K);
DP(:,setf(:,1)) = D_hat;
for g = 1:N_RF
    userg = setf(g,:);  % 第n个beam的用户
    userg(userg==0) = [];
    for ng = 2:length(userg)
        DP(:,userg(ng)) = DP(:,setf(g,1));
    end
end




