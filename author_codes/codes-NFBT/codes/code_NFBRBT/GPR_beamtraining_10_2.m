function [mu,cor,index_A,h_o,kmax] = GPR_beamtraining_10_2(Kernal,sigma2,Observe_step,S,max_GPR_iter,QUA,Gain_vector)
index_A = [];                % Index of observed points
A_t = [];                    % Observed points
h_o = [];                    % Value of observed points



mu = zeros(S,1);
cor = ones(S,1);

Candidate_index = nchoosek(1:1:S,Observe_step);

for aa = 1:max_GPR_iter
    if aa == 1
        kmax = randperm(S,Observe_step);
      

    else
        Candidate_num = size(Candidate_index,1);
        Candidate_value = zeros(Candidate_num,1);
        mu_temp = zeros(Candidate_num,1);


        [h_o_max_value,h_o_max_index] = max(h_o);

        for bb =1:Candidate_num
            mu_temp(bb) = mu(Candidate_index(bb,:));
            Candidate_value(bb) = norm(mu_temp(bb)-h_o_max_value)^2;
        end

   
        [~,kmax_index] = min(Candidate_value);
        kmax = Candidate_index(kmax_index,:);
    end



    for bb = 1:Observe_step
        [Remove_index,~] = find(Candidate_index==kmax(bb));
        Candidate_index(Remove_index,:) = [];
    end

    A_t = [A_t QUA(kmax,:)];

    % h_o = [h_o; abs(Gain_vector(kmax) + sqrt(sigma2)*(randn(Observe_step,1)+1j*randn(Observe_step,1))/sqrt(2))];
    h_o = [h_o; abs(Gain_vector(kmax))];

    index_A = [index_A kmax];

    Observed_length = length(h_o);
    k_t = zeros(Observed_length,1);
    K_t = Kernal(index_A,index_A);

    k_t = Kernal(index_A,:);
    mu = k_t'*(K_t)^(-1)*(h_o);
    Sigma_k = Kernal - k_t'*(K_t)^(-1)*k_t;
    cor = abs(sqrt(diag(Sigma_k)));
 

end