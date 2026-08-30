function [mu,cor,index_A,h_o] = GPR_multi(x,h,Kernal,sigma2,P,Observe_step)

N = length(x);

index_A = [];                % Index of observed points
A_t = [];                    % Observed points
h_o = [];                    % Value of observed points

% Observe_step = 3;            % Number of observed points in each iteration

mu = zeros(N,1);
cor = ones(N,1);

Candidate_index = nchoosek(1:1:N,Observe_step);

for aa = 1:P

    if aa ==1
        % kmax = randi(N);
        kmax = randperm(N,Observe_step);
        % kmax = [1 2];

    else
        Candidate_num = size(Candidate_index,1);
        Candidate_value = zeros(Candidate_num,1);

        for bb =1:Candidate_num
            Cov_temp = Sigma_k(Candidate_index(bb,:),Candidate_index(bb,:));
            Candidate_value(bb) = abs(det(Cov_temp + sigma2*eye(Observe_step)));
        end
        [~,kmax_index] = max(Candidate_value);
        kmax = Candidate_index(kmax_index,:);
    end

    for bb = 1:Observe_step
        [Remove_index,~] = find(Candidate_index==kmax(bb));
        Candidate_index(Remove_index,:) = [];
    end

    A_t = [A_t x(kmax)];

    h_o = [h_o; h(kmax) + sqrt(sigma2)*(randn(Observe_step,1)+1j*randn(Observe_step,1))/sqrt(2)];

    index_A = [index_A kmax];

    Observed_length = length(h_o);
    k_t = zeros(Observed_length,1);
    K_t = Kernal(index_A,index_A);

    k_t = Kernal(index_A,:);
    mu = k_t'*(K_t+sigma2*eye(Observed_length))^(-1)*h_o;
    Sigma_k = Kernal - k_t'*(K_t+sigma2*eye(Observed_length))^(-1)*k_t;
    cor = abs(sqrt(diag(Sigma_k)));
end

