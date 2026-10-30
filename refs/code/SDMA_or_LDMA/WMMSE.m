function [D_opt] = WMMSE(H, K, PK, sigma2, V_init)
%WMMSE

iter = 50;
vnew = 0;
epsilon = 1e-2; 
break_epsilon = epsilon;
step = 0;

%% initialization
% V = sqrt(PK/K)*ones(K,K);
V = V_init;
U = zeros(K,1);
W = U;

while break_epsilon >= epsilon && step <= 50
    step = step +1;
    Keep_W = W;

    U = zeros(K,1);
    W = U;
    for i = 1:K
        J = 0;
        for j = 1:K
            J = J + H(j,:)*V(:,j)*V(:,j)'*H(j,:)';
        end
        J = J+sigma2;
        U(i) = J^(-1)*H(i,:)*V(:,i);
        W(i) = 1/(1-V(:,i)'*H(i,:)'*J^(-1)*H(i,:)*V(:,i));
        vnew = vnew + log2(W(i));
    end
    

    M = zeros(K,K);
    for mk = 1:K
        M = M + H(mk,:)'*U(mk)*W(mk)*U(mk)'*H(mk,:);
    end
    


    p = 0;
    flag = 0;
    if (abs(det(M)) > 1e-30)
        for mk = 1:K
            V(:,mk) = M\H(mk,:)'*U(mk)*W(mk);
            p = p + trace(V(:,mk)*V(:,mk)');
        end
        if p<=PK
            mu = 0;
            flag = 1;
        end
    end
    if flag ~= 1
        [D, A] = eig(M);
        N = zeros(K,K);
        for mk = 1:K
            N= N + H(mk,:)'*U(mk)*W(mk)*W(mk)*U(mk)'*H(mk,:);
        end
        Phi = D' * N * D;
        
        mu_high = 100;
        mu_low = 0;
        while abs(p - PK)>=1e-5
            p = 0;
            mu = (mu_high + mu_low)/2;
            for t = 1:K
                p = p + Phi(t,t)/((A(t,t) + mu)^2);
            end
            
            if p > PK
                mu_low = mu;
            elseif p < PK
                mu_high = mu;
            end
        end
        for mk = 1:K
            V(:,mk) = (M+mu*eye(K))\H(mk,:)'*U(mk)*W(mk);
        end
    end
    

    break_epsilon = 0;
    for ik = 1:K
        break_epsilon = break_epsilon + log(det(Keep_W(ik))) - log(det(W(ik)));
    end
    break_epsilon = abs(break_epsilon);

end

D_opt = V;
end