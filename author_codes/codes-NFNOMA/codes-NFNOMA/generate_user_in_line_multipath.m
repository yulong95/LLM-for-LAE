function [H_multi_user] = generate_user_in_line_multipath(Nt, d, r_min, r_max, f, N_user, sigma_aod, L, kappa)
%GENERATE_USER_IN_LINE_MULTIPATH 此处显示有关此函数的摘要
%   此处显示详细说明

    c = 3e8;
    H_multi_user = zeros(N_user, Nt);
    
    %% uniform random in line
    theta = rand(1,1)*2*pi/3-pi/3;
    r_list = rand(1, N_user)*(r_max-r_min) + r_min;
    theta_list = theta*ones(1, N_user);
    
    nn = -(Nt-1)/2:1:(Nt-1)/2;
    theta_aod  = sqrt(sigma_aod)*randn(N_user,L);
    ssf = (randn(N_user,L) + 1j*randn(N_user,L))/sqrt(2);
    % allocate a factor to fade the NLoS channel
    alpha = 1;
    beta = sqrt(alpha/(L));
    ssf = ssf*beta;
    
    for i_user = 1:N_user
        for l = 1:L+1
            if l ~= L+1
                r0 = r_list(i_user);
                theta0 = theta_list(i_user)+theta_aod(i_user,l);
                r = sqrt(r0^2 + (nn*d).^2 - 2*r0*nn*d*sin(theta0));
                at = exp(-1j*2*pi*f*(r - r0)/c)/sqrt(Nt);
                H_multi_user(i_user, :) = H_multi_user(i_user, :) + ssf(i_user,l)*at*sqrt(1/(1+kappa));
            else
                r0 = r_list(i_user);
                theta0 = theta_list(i_user);
                r = sqrt(r0^2 + (nn*d).^2 - 2*r0*nn*d*sin(theta0));
                at = exp(-1j*2*pi*f*(r - r0)/c)/sqrt(Nt);
                H_multi_user(i_user, :) = H_multi_user(i_user, :) + at*sqrt(kappa/(1+kappa));
            end
        end
    end
    
    if L ==0
        H_multi_user = H_multi_user/sqrt(kappa/(1+kappa));
    end

end

