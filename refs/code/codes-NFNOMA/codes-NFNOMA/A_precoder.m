function [H_eq_full,AP_full,H_eq_sub,AP_sub,setu] = A_precoder(H,N,N_RF,K,bit)

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

%% analog precoding - sub
Nr = N/N_RF;            % number of antennas for each RF chain
A_hat = zeros(Nr,N_RF);
AP_sub = zeros(N,N_RF);     % analog precoding
N_phase = 2^bit;
phase_set = (2*pi/N_phase)*[0:N_phase-1] - pi;
for g = 1:N_RF
    Hs = H(Nr*(g-1)+1:Nr*g,setu(g));
    phase = angle(Hs);
    Q_phase = zeros(Nr,1);
    for n = 1:Nr
        [~,i] = min(abs(phase(n) - phase_set));
        Q_phase(n) = phase_set(i);
    end
    A_hat(:,g) = exp(1i*Q_phase);
end
for g = 1:N_RF
    AP_sub(Nr*(g-1)+1:Nr*g,g) = A_hat(:,g);
end

H_eq_sub = H'*AP_sub;

%% analog precoding - full
AP_full = zeros(N,N_RF);     % analog precoding
N_phase = 2^bit;
phase_set = (2*pi/N_phase)*[0:N_phase-1] - pi;
for g = 1:N_RF
    Hs = H(:,setu(g));
    phase = angle(Hs);
    Q_phase = zeros(N,1);
    for n = 1:N
        [~,i] = min(abs(phase(n) - phase_set));
        Q_phase(n) = phase_set(i);
    end
    AP_full(:,g) = exp(1i*Q_phase);
end

H_eq_full = H'*AP_full;



