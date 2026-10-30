function F_RF = PEAlt(F_opt, N_t, N_RF, N_s)
   F_RF = exp(1j*pi*(rand(N_t, N_RF)*2-1));
   
   M = size(F_opt,3);
%    F_BB = zeros(N_RF, N_s, M);
   F_DD = zeros(N_RF, N_s, M);
   N_iter = 10;
   
   for idx = 1:N_iter
       for idx_M = 1:M
            [U, ~, V] = svd(F_opt(:,:,idx_M)'*F_RF);
            V1 = V(:,1:N_s);
            F_DD(:,:,idx_M) = V1*U';
       end
       F_temp = zeros(size(F_RF));
       for idx_M = 1:M
           F_temp =  F_temp + F_opt(:,:,idx_M)*F_DD(:,:,idx_M)';
       end
       F_RF = F_temp./abs(F_temp);
   end
   
   F_BB = zeros(N_RF,N_s,M);
   for idx_M = 1:M
        F_BB(:,:,idx_M) = F_RF\F_opt(:,:,idx_M);
   end

end

function l = loss(F_opt, F_RF, F_BB)
    M = size(F_opt,3);
    l = 0;
    for idx = 1:M
       l = l+norm(F_opt(:,:,idx) - F_RF*F_BB(:,:,idx), 'F')^2;
    end
    l = l/M;
end

