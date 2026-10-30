function [at_real, at_wo] = near_field_3D(Nt1,Nt2, d, f, r0, theta0, phi0)
    c = 3e8;
    nn1 = -(Nt1-1)/2:1:(Nt1-1)/2;
    nn2 = (-(Nt2-1)/2:1:(Nt2-1)/2)';
    
    r_wo = zeros(Nt1, Nt2);
    r_real = zeros(Nt1, Nt2);
    
    for i_iter = 1:length(nn1)
        for j_iter = 1:length(nn2)
            r_real(i_iter, j_iter) = sqrt(r0^2-2*r0*nn1(i_iter)*d*cos(theta0)-2*r0*nn2(j_iter)*d*sin(theta0)*sin(phi0)...
                +nn1(i_iter)^2*d^2+nn2(j_iter)^2*d^2);
            
            r_wo(i_iter, j_iter) = (r0 - nn1(i_iter)*d*cos(theta0) - nn2(j_iter)*d*sin(theta0)*sin(phi0) +...
                (nn1(i_iter)*d)^2*(1-cos(theta0)^2)/2/r0 + (nn2(j_iter)*d)^2*(1-(sin(theta0)^2)*(sin(phi0)^2))/2/r0);
        end
    end
    at_wo = exp(-1j*2*pi*f*(r_wo - r0)/c)/sqrt(Nt1)/sqrt(Nt2);
    at_wo = reshape(at_wo, 1, []);
    
    at_real = exp(-1j*2*pi*f*(r_real - r0)/c)/sqrt(Nt1)/sqrt(Nt2);
    at_real = reshape(at_real, 1, []);
    
end

