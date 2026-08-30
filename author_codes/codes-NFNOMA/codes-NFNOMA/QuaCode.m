function [w, sin_theta_list, r_list] = QuaCode(Nt, d, lambda_c, delta, rho_min)
    c = 3e8;
    fc = c/lambda_c;
    sin_theta_list = -1 + 2/Nt : 2/Nt : 1 - 2/Nt;
    D = size(sin_theta_list, 2);
    
    Z_delta = (Nt * d)^2 / 2 / lambda_c / delta^2;
    S = floor(Z_delta / rho_min) + 1;
    r_list = zeros(S, 1);
    for s = 1:S
        if s == 1
            r_list(s) = 200 * Nt^2 * d^2 / lambda_c;
        else
            r_list(s) = Z_delta / (s - 1);
        end
    end
    
    %% narrowband polar-domain codebook
    w = zeros(Nt, D, S);
    for s = 1:S
        for idx = 1:D
            sin_theta = sin_theta_list(idx);
            r = r_list(s)*(1 - sin_theta^2);
            w(:, idx, s) = polar_domain_manifold( Nt, d, fc, r, asin(sin_theta) );
        end
    end
end

function  at = polar_domain_manifold( Nt, d, f, r0, theta0 )
    c = 3e8;
    nn = -(Nt-1)/2:1:(Nt-1)/2;
%     r = sqrt(r0^2 + (nn*d).^2 - 2*r0*nn*d*sin(theta0));
    r = r0 - nn * d * sin(theta0) + nn.^2 * d^2 * cos(theta0)^2 ./ 2 /r0;
    at = exp(-1j*2*pi*f*(r-r0)/c)/sqrt(Nt);
    at = at.';
end

