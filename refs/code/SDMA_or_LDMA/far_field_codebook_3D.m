function [DFT_t, theta_grid, phi_grid] = far_field_codebook_3D(Nt1, Nt2, d, lambda)
%FAR_FIELD_CODEBOOK_3D 此处显示有关此函数的摘要
%   此处显示详细说明
%     sin_theta = -1+1/Nt1:2/Nt1:1-1/Nt1;
    sin_theta = -sqrt(3)/2:2/Nt1:sqrt(3)/2;
    theta = asin(sin_theta);

%     sin_gamma =  -1+1/Nt2:2/Nt2:1-1/Nt2;
    sin_gamma = -sqrt(3)/2:2/Nt1:sqrt(3)/2;
    gamma = asin(sin_gamma);

    [theta_grid, gamma_grid] = meshgrid(theta, gamma);
    phi_grid = asin(sin(gamma_grid)./cos(theta_grid));
    
    nn1 = -(Nt1-1)/2:1:(Nt1-1)/2;
    nn2 = (-(Nt2-1)/2:1:(Nt2-1)/2)';

    DFT_t = zeros(Nt1*Nt2, length(theta), length(gamma));
    for iter=1:length(theta)
        theta0 = theta(iter) + pi/2;
        for iter2 = 1:length(gamma)
            phi0 = phi_grid(iter2, iter);
            if isreal(phi0)
                r_far = zeros(Nt1, Nt2);
                for i_idx = 1:length(nn1)
                    for j_idx = 1:length(nn2)
                        r_far(i_idx, j_idx) = (-nn1(i_idx)*d*cos(theta0) - nn2(j_idx)*d*sin(theta0)*sin(phi0));
                    end
                end
                at_far = exp(-1j*2*pi/lambda*r_far)/sqrt(Nt1)/sqrt(Nt2);
                DFT_t(:, iter, iter2) = reshape(at_far, [], 1);
            end
        end
    end
    DFT_t = reshape(DFT_t, Nt1*Nt2, []);
    
end

