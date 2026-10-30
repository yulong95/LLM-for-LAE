function [F_cal] = cal_zero_forcing(H_effect, precoding)
%CAL_ZERO_FORCING 此处显示有关此函数的摘要
%   此处显示详细说明
    F_cal = pinv(H_effect);
    F_cal = F_cal/norm(precoding*F_cal, 'fro');
    F_near_LS_power = norm(precoding*F_cal, 'fro');
end

