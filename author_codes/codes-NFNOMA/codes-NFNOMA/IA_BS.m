function [Hr,Hr_e] = IA_BS(H_est,H)
[n,K] = size(H);
s_set = 1:n;
deta = 10^-1;
% H£ºbeamspace channel matrix (sparse)
% N; number of retained beams 
% n: total number of beams (transmit antennas)
H_abs = abs(H_est);
r = [];
m = [];
for k = 1:K
    [~,m(k)] = max(H_abs(:,k));
end
m_set = unique(m);
b_set = m_set; 
for i = 1:length(m_set)
    flag = find(m==m_set(i));
    if length(flag) > 1;
        b_set(b_set==m_set(i)) = [];
    end
end
s_set(b_set)=[];
no = K - length(b_set);
if no~=0
   T_set = b_set;
   %%% proposed search 
   for i = 1:no
       A = H_est(T_set,:);
       B = inv(A'*A + deta*eye(K,K));
       target = [];
      for j = 1:length(s_set)
          temp = H_est(s_set(j),:);
          target(j) = (temp*B^2*temp')/(1+temp*B*temp');
      end
      [~,jj] = max(target);
      T_set = unique([s_set(jj) T_set]);
      s_set(jj) = [];
   end
   %%% exhaustive search
%    for i = 1:size(search_set,1)
%        set = search_set(i,:);
%        T_set = unique([set b_set]);
%        H_r = H(T_set,:);
%        F = H_r*inv(H_r'*H_r);
%        beta(i) = sqrt(K/trace(F'*F));
%    end
%    [~,ii] = max(beta);
%    T_set_opt = unique([search_set(ii,:) b_set]);
else
   T_set = b_set;
end
Hr =  H(T_set,:);
Hr_e = H_est(T_set,:);