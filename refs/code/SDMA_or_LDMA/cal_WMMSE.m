function [F_MMSE] = cal_WMMSE(H_e_perfect, precoding, W, N_1, N_user, N_ris, Power_total, sigma2)
%CAL_WMMSE 此处显示有关此函数的摘要
%   此处显示详细说明

    [F_MMSE, ~] = NoRIS_precoding(N_1, N_user,...
        N_ris, Power_total, sigma2, ones(N_user,1), conj(W), H_e_perfect,...
        zeros(N_user,N_ris), zeros(N_ris,N_1));
    F_MMSE = conj(reshape(F_MMSE, [N_1,N_user]));
    F_MMSE = F_MMSE/norm(precoding*F_MMSE, 'fro');
    FMMSE_power = norm(precoding*F_MMSE, 'fro');

end

