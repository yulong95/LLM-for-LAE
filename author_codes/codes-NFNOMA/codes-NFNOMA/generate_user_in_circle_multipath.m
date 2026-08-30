function [H_multi_user] = generate_user_in_circle_multipath(Nt, d, r_min, r_max, f, N_user, sigma_aod, L, kappa)

    c = 3e8;
    H_multi_user = zeros(N_user, Nt);
    
    %% uniform random
    cnt = 1;
    x_list = zeros(1,N_user);
    y_list = zeros(1,N_user);
    while(cnt <= N_user)
        x_1 = abs(rand()*r_max);
        y_1 = (rand()-0.5)*2*r_max;
        [theta_1, r_1]  = cart2pol(x_1, y_1);
        if (x_1^2+y_1^2<r_max^2) && (x_1^2+y_1^2>r_min^2) && (theta_1>-pi/6) && (theta_1<pi/6)
            x_list(cnt) = x_1;
            y_list(cnt) = y_1;
            cnt = cnt+1;
        end
    end
    [theta_list, r_list]  = cart2pol(x_list, y_list);
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

