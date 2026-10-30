function [w, theta_record_list, phi_record_list, r_list] = Code3D(Nt1, Nt2, d, lambda, delta, rho_min)

num_point = 1000;
c = 3e8;
eps_est = 2e-3;
eps = 1e-3;
fc = c/lambda;
max_r = 10;
beta_list = linspace(0.0001,0.05,2000);
theta_list = asin(-sqrt(3)/2:2/Nt1:sqrt(3)/2);
gamma_list = asin(-sqrt(3)/2:2/Nt2:sqrt(3)/2);
% theta_list = asin(-1/12:2/Nt1:1/12);
% gamma_list = asin(-1/12:2/Nt2:1/12);

r_delta = zeros(length(gamma_list),length(theta_list));
[theta_grid, gamma_grid] = meshgrid(theta_list, gamma_list);
phi_grid = asin(sin(gamma_grid)./cos(theta_grid));



beta_idx_list = zeros(length(gamma_list),length(theta_list));
theta_record_list = zeros(1, length(gamma_list)*length(theta_list)*max_r);
phi_record_list = zeros(1, length(gamma_list)*length(theta_list)*max_r);
r_list = zeros(1, length(gamma_list)*length(theta_list)*max_r);
w = zeros(Nt1*Nt2, length(gamma_list)*length(theta_list)*max_r);

cnt = 1;
for i_theta = 1:length(theta_list)
    for i_gamma = 1:length(gamma_list)
        theta = theta_grid(i_gamma, i_theta);
        phi = phi_grid(i_gamma, i_theta);
        if isreal(phi)
            G = zeros(1,length(beta_list));
            for i = 1:length(beta_list)
                beta0 = beta_list(i);
                beta1 = beta0*sqrt(1-sin(theta)^2)*Nt1;
                beta2 = beta0*sqrt(1-cos(theta)^2*sin(phi)^2)*Nt2;
                [f1_sin, f1_cos] = fresnel(beta1, num_point);
                [f2_sin, f2_cos] = fresnel(beta2, num_point);
                G(i) = abs((f1_cos+1i*f1_sin)/beta1 * (f2_cos+1i*f2_sin)/beta2);
            end
            % find the corresponding beta for any delta
            beta_err = G-delta;
            [est_err, beta_idx] = min(abs(beta_err));
            if  est_err > eps_est
                fprintf('beta is not accurate, err is %4.4f \n',est_err);
            end
            beta_delta = beta_list(beta_idx);
            beta_idx_list(i_gamma, i_theta) = beta_idx;
            r_delta(i_gamma,i_theta) = d^2/beta_delta^2/2/lambda;

            Z_delta = r_delta(i_gamma,i_theta);
            S = floor((Z_delta+eps) / rho_min) + 1;
            if S > 8
                bbb = 1;
            end

            for s = 1:S
                if s == 1
                    distance = 200 * Nt1^2 * d^2 / lambda;
                    r_list(cnt) = distance;
                    [at_real, at_wo] = near_field_3D(Nt1,Nt2, d, fc, distance, pi/2-theta, phi);
                    w(:, cnt) = at_wo;
                    theta_record_list(cnt) = theta;
                    phi_record_list(cnt) = phi;
                    cnt = cnt+1;
                else
                    distance = Z_delta / (s - 1);
                    r_list(cnt) = distance;
                    [at_real, at_wo] = near_field_3D(Nt1,Nt2, d, fc, distance, pi/2-theta, phi);
                    w(:, cnt) = at_wo;
                    theta_record_list(cnt) = theta;
                    phi_record_list(cnt) = phi;
                    cnt = cnt+1;
                end
            end
        end
    end
end
w = w(:,1:cnt-1);
theta_record_list = theta_record_list(:,1:cnt-1);
phi_record_list = phi_record_list(:,1:cnt-1);
r_list = r_list(:,1:cnt-1);

end

