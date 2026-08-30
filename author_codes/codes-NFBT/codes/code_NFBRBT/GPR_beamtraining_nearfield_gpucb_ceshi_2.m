%% gpr_success
function [mu,cor,index_A,h_o,kmax] = GPR_beamtraining_nearfield_gpucb_ceshi_2(Kernal,sigma2,Observe_step,S,max_GPR_iter,Gain_vector)
index_A = [];                % Index of observed points
% A_t = [];                    % Observed points
h_o = [];                    % Value of observed points

% beta = 1;
delta = 0.1;
taut = zeros(max_GPR_iter,1);
nu = 1;

mu = zeros(S,1);
cor = ones(S,1);

Candidate_index = nchoosek(1:1:S,Observe_step);

for aa = 1:max_GPR_iter
    if aa == 1
        kmax = randperm(S,Observe_step);

    else
        Candidate_num = size(Candidate_index,1);
        Candidate_value = zeros(Candidate_num,1);
        Cov_temp = zeros(Candidate_num,1);
        mu_temp = zeros(Candidate_num,1);
        cor_temp = zeros(Candidate_num,1);

       % [h_o_max_value,h_o_max_index] = max(h_o);


        % taut(aa) = 2*log(((aa)^(1/2+2))*(pi^2/(3*delta)));
        taut(aa) = 2*log((max_GPR_iter*aa^2*pi^2)/(6*delta));
        % taut(aa) = (2/5)*log((max_GPR_iter*aa^2*pi^2)/(6*delta));


        for bb =1:Candidate_num
            Cov_temp(bb) = Sigma_k(Candidate_index(bb,:),Candidate_index(bb,:));
            mu_temp(bb) = mu(Candidate_index(bb,:));
            cor_temp(bb) = cor(Candidate_index(bb,:));


            Candidate_value(bb) = abs(Cov_temp(bb));
            % Candidate_value(bb) = abs(mu_temp(bb)) + beta * abs(Cov_temp(bb));
            % if aa >= max_GPR_iter*(2/3)
            %     Candidate_value(bb) = abs(mu_temp(bb));
            %     % Candidate_value(bb) = 1/(norm(mu_temp(bb)-h_o_max_value)^2);
            % else
            %     Candidate_value(bb) =  abs(Cov_temp(bb));
            % end

            % Candidate_value(bb) =  abs(Cov_temp(bb));
            % Candidate_value(bb) = abs(mu_temp(bb));
        end
        % [Cov_temp_max,Cov_temp_index] = max(Cov_temp);
        % [cor_temp_max,cor_temp_index] = max(cor_temp);
        % [mu_temp_max,mu_temp_index] = max(mu_temp);
        
        % if Cov_temp_max <= 0.85
        %    Candidate_value = mu_temp;
        % else
        %    Candidate_value = Cov_temp;
        % end
        % xi = 0.01;
        % [ei, maxEIIndex] = expectedImprovement(mu_temp, cor_temp, h_o_max_value, xi);
        % [ei, maxEIIndex] = expectedImprovement_2(mu_temp, cor_temp, h_o_max_value, xi,Candidate_num);

        [Candidate_value_max,kmax_index] = max(Candidate_value);
        kmax = Candidate_index(kmax_index,:);
    end
        


    for bb = 1:Observe_step
        [Remove_index,~] = find(Candidate_index==kmax(bb));
        Candidate_index(Remove_index,:) = [];
    end

    
    % A_t = [A_t QUA(kmax,:)];

    h_o = [h_o; Gain_vector(kmax)];

    index_A = [index_A kmax];

    Observed_length = length(h_o);
    k_t = zeros(Observed_length,1);
    K_t = Kernal(index_A,index_A);

    k_t = Kernal(index_A,:);
    mu = k_t'*(K_t)^(-1)*h_o;
    Sigma_k = Kernal - k_t'*(K_t)^(-1)*k_t;
    cor = abs(sqrt(diag(Sigma_k)));

end