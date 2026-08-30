function [H_eq_full,AP_full,setu] = A_precoder_NF(H , N , N_RF , K , w_near , num_theta , num_dis)
%%
% [H_eq_full,AP_full,setu] = A_precoder_NF(H_mul_user_transpose , n , N , K , w_near , num_theta , num_dis);
% H：输入的信道满足n*K,即H_mul_user_transpose
% N:天线数
% N_RF:射频链数
% K:用户数
%%
Hnorm = sqrt(sum(abs(H).^2,1));  % channel gains for each user
Ht = H./repmat(Hnorm,N,1);       % normalization for each user's channel
[~,order] = sort(Hnorm,'descend');
rho = 0.3;                       % correlation threshold

%% generate user set for analog precoding
setu = zeros(N_RF,1);  % selected user set with low correlation
if K<=N_RF
    setu(1:K) = 1:K;
else
    setu(1) = order(1);
    order(1) = [];
    setg = order;   % remaining user set
    g = 2;
    while g <= N_RF
        if isempty(setg)
            while isempty(setg)
                setg = order;
                rho = rho+(1-rho)/10;  % adjust correlation threshold
                x1 = abs(Ht(:,setu(1:g-1))'*Ht(:,setg));
                x2 = max(x1,[],1);
                x3 = find(x2>=rho);
                setg(x3) = [];
            end
        end
        corr = abs(Ht(:,setu(g-1))'*Ht(:,setg));
        ind = find(corr<rho);
        if isempty(ind)
            setg = [];
            continue;
        else
            setu(g) = setg(ind(1));
            order(find(order==setu(g))) = [];
            if length(ind)>1
                setg = setg(ind(2:end));
            else
                setg = [];
            end            
        end
        g = g+1;
    end
end

%% analog precoding_1_通过beamtraining_整体
% for g = 1:N_RF
%     Hs = H(:,setu(g)); % 从n*K变成n*N (其中n代表基站发射天线数，N代表射频链数目即N_RF)
% end
% Hs_transpose = Hs.'; % N*n 即射频链数*基站发射天线数
% [precoding_matrix_near_near_Hs] = select_beam_near(Hs_transpose, w_near, N, N_RF, num_theta, num_dis); % n*N 即基站发射天线数*射频链数
% AP_full = precoding_matrix_near_near_Hs; % analog precoding,N*N_RF即n*N

%% analog precoding_2_通过beamtraining_逐个
AP_full = zeros(N,N_RF);
for g = 1:N_RF
    Hs = H(:,setu(g)); % 从n*K变成n*N (N = 1)(其中n代表基站发射天线数，N代表射频链数目即N_RF)
    Hs_transpose = Hs.'; % N*n (N = 1) 射频链数*基站发射天线数
    [precoding_matrix_near_near_Hs] = select_beam_near(Hs_transpose, w_near, N, 1, num_theta, num_dis); % n*N (N = 1) 即基站发射天线数*射频链数
    AP_full(:,g) = precoding_matrix_near_near_Hs;
end

%% 等效信道
H_eq_full = H'*AP_full; % (K*n)*(n*N) = K*N即用户数*射频链数
% H_eq_full = H.'*AP_full; % (K*n)*(n*N) = K*N即用户数*射频链数



