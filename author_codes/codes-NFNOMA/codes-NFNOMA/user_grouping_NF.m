function setf = user_grouping_NF(H_eq,N_RF,K,setu)
% setf3_7 = user_grouping_NF(HA_full,N,K,setu); % N*K
Hnorm = sqrt(sum(abs(H_eq).^2,2));     % channel gains for each user
Ht = H_eq./repmat(Hnorm,1,N_RF);       % normalization for each user's channel

setf = zeros(N_RF,K);  % user group
setf(:,1) = setu;
setr = setdiff([1:K],setu);

x1 = abs(Ht(setu,:)*Ht(setr,:)');
[~, x3] = max(x1,[],1);
for g = 1:N_RF
    i = find(x3==g);
    num_userg = length(i)+1;
    userg = [setf(g,1), setr(i)];
    setf(g,1:num_userg) = userg;
    [~,j] = sort(Hnorm(userg),'descend');
    setf(g,1:num_userg) = userg(j);
end
        
    
    