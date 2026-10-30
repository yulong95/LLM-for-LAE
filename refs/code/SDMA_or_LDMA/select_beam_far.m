function [precoding_matrix_near_far] = select_beam_far(H_mul_user, DFT_t, Nt, N_user)
%SELECT_BEAM_FAR 此处显示有关此函数的摘要
%   此处显示详细说明
    beam_gain = abs(H_mul_user*DFT_t);
    [~, idx_max] = max(beam_gain, [], 2);
    precoding_matrix_near_far = DFT_t(:,idx_max);
end

