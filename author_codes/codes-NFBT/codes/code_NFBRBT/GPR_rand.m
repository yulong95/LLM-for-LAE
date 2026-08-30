function [mu,cor,index_A,h_o] = GPR_rand(x,h,Kernal,sigma2,P)

N = length(x);

index_A = [];                % Index of observed points
A_t = [];                    % Observed points
h_o = [];                    % Value of observed points

% Observe_step = 3;            % Number of observed points in each iteration

mu = zeros(N,1);
cor = ones(N,1);

index_A = sort(randperm(N,P));
A_t = x(index_A);

h_o = h(index_A) + sqrt(sigma2)*(randn(P,1)+1j*randn(P,1))/sqrt(2);

Observed_length = length(h_o);
k_t = zeros(Observed_length,1);
K_t = Kernal(index_A,index_A);

k_t = Kernal(index_A,:);

mu = k_t'*(K_t+sigma2*eye(P))^(-1)*h_o;
Sigma_k = Kernal - k_t'*(K_t+sigma2*eye(Observed_length))^(-1)*k_t;
cor = abs(sqrt(diag(Sigma_k)));

end

