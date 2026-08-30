function [mu,cor,index_A,h_o,kmax,rate_near_GPR] = GPR_beamtraining_nearfield_gpucb_ceshi_rate_vs_overhead(Kernal,sigma2,Observe_step,S,max_GPR_iter,Gain_vector,sample,t,SNR)
%%
rate_near_GPR = zeros(sample,S);
%% 
index_A = [];                % Index of observed points
h_o = [];                    % Value of observed points
mu = zeros(S,1);
cor = ones(S,1);
Candidate_index = nchoosek(1:1:S,Observe_step);
%%

array_gain_GPR_near = 0;
for aa = 1:max_GPR_iter
    fprintf('t = [%d/%d] |aa = [%d/%d]\n',t, sample,aa,max_GPR_iter); 
    %%
    if aa == 1
        kmax = randperm(S,Observe_step);
    else
        Candidate_num = size(Candidate_index,1);
        Candidate_value = zeros(Candidate_num,1);
        Cov_temp = zeros(Candidate_num,1);
        mu_temp = zeros(Candidate_num,1);
        cor_temp = zeros(Candidate_num,1);

        for bb =1:Candidate_num
            Cov_temp(bb) = Sigma_k(Candidate_index(bb,:),Candidate_index(bb,:));
            mu_temp(bb) = mu(Candidate_index(bb,:));
            cor_temp(bb) = cor(Candidate_index(bb,:));
            Candidate_value(bb) = abs(Cov_temp(bb));
        end
        [Candidate_value_max,kmax_index] = max(Candidate_value);
        kmax = Candidate_index(kmax_index,:);
    end

    for bb = 1:Observe_step
        [Remove_index,~] = find(Candidate_index==kmax(bb));
        Candidate_index(Remove_index,:) = [];
    end

    h_o = [h_o; Gain_vector(kmax)];

    index_A = [index_A kmax];

    Observed_length = length(h_o);
    k_t = zeros(Observed_length,1);
    K_t = Kernal(index_A,index_A);

    k_t = Kernal(index_A,:);
    mu = k_t'*(K_t)^(-1)*h_o;
    Sigma_k = Kernal - k_t'*(K_t)^(-1)*k_t;
    cor = abs(sqrt(diag(Sigma_k)));
    %% first round search
    mu_baseline = 0;
    for i = 1:S
       if mu_baseline <= abs(mu(i))
          mu_baseline = abs(mu(i));
          GPR_index = i;
       end
    end
  
    %% fine search begin
    Gain_baseline = 0;
    if (GPR_index >= 20) && (GPR_index <= (S-20))

    for i = (GPR_index-19):(GPR_index+20)
        if Gain_baseline <= abs(Gain_vector(i))^2
           Gain_baseline = abs(Gain_vector(i))^2;
           GPR_index = i;
       end
    end

    elseif (GPR_index >= 1) && (GPR_index <= 19)

    for i = 1:20
        if Gain_baseline <= abs(Gain_vector(i))^2
           Gain_baseline = abs(Gain_vector(i))^2;
           GPR_index = i;
       end
    end

    elseif (GPR_index <= S) && (GPR_index >= (S-19))

    for i = (S-19):S
       if Gain_baseline <= abs(Gain_vector(i))^2
           Gain_baseline = abs(Gain_vector(i))^2;
           GPR_index = i;
       end
    end


    end
    % fine search end

    % fine search 2 begin

    for i = 1:aa
       if Gain_baseline <= abs(h_o(i))^2
          Gain_baseline = abs(h_o(i))^2;
          GPR_index = index_A(i);
       end
    end

    % fine search 2 end
    %%



   %%

   %%
   if array_gain_GPR_near <= (abs(Gain_vector(GPR_index,1)))^2
      array_gain_GPR_near = max(array_gain_GPR_near,(abs(Gain_vector(GPR_index,1)))^2);
      rate_near_GPR(t,aa) = log2(1 + SNR * array_gain_GPR_near);
   else
      rate_near_GPR(t,aa) = rate_near_GPR(t,aa-1);
   end

end
%%