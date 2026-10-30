function [precoding_matrix_near_far] = select_beam_far_3D(H_mul_user, DFT_t, Nt1, Nt2, N_user)

    
    beam_gain = abs(H_mul_user*(DFT_t));
    [max_value, idx_max] = max(beam_gain, [], 2);
    precoding_matrix_near_far = DFT_t(:,idx_max);
    
end

