function [sum_rate_reture] = cal_sum_rate_mu(H_mul_user, F_precoding, N_user, sigma2)
%CAL_SUM_RATE_MU 此处显示有关此函数的摘要
%   此处显示详细说明
    c_near_near = zeros(1,N_user);
    gain_near_near = abs(H_mul_user*F_precoding).^2;
    for i_user = 1:N_user
        signal_power = gain_near_near(i_user,i_user);
        interference_power = sum(gain_near_near(i_user,:)) - gain_near_near(i_user,i_user);
        c_near_near(i_user) = log2(1+signal_power/(sigma2+interference_power));
    end
    sum_rate_reture = sum(c_near_near);
end

