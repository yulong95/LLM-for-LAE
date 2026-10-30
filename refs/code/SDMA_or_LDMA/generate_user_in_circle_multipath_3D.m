function [H_multi_user] = generate_user_in_circle_multipath_3D(Nt1, Nt2, d, r_min, r_max, fc, N_user, sigma_aod, L, kappa)

    c = 3e8;
    H_multi_user = zeros(N_user, Nt1*Nt2);
    
    %% uniform random in line
    theta_list = rand(1,N_user)*2*pi/6-pi/6+pi/2;
    phi_list = rand(1,N_user)*2*pi/6-pi/6;
    r_list = rand(1, N_user)*(r_max-r_min) + r_min;
    theta_aod  = sqrt(sigma_aod)*randn(N_user,L);
    phi_aod  = sqrt(sigma_aod)*randn(N_user,L);
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
                phi0 = phi_list(i_user)+phi_aod(i_user,l);
                [at_real, at_wo] = near_field_3D(Nt1,Nt2, d, fc, r0, theta0, phi0);
                H_multi_user(i_user, :) = H_multi_user(i_user, :) + ssf(i_user,l)*at_real*sqrt(1/(1+kappa));
            else
                r0 = r_list(i_user);
                theta0 = theta_list(i_user);
                phi0 = phi_list(i_user);
                [at_real, at_wo] = near_field_3D(Nt1,Nt2, d, fc, r0, theta0, phi0);
                H_multi_user(i_user, :) = H_multi_user(i_user, :) + at_real*sqrt(kappa/(1+kappa));
            end
        end
    end
    
    if L ==0
        H_multi_user = H_multi_user/sqrt(kappa/(1+kappa));
    end
    

end

