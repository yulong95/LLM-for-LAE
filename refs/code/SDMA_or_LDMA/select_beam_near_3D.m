function [precoding_matrix_near_near] = select_beam_near_3D(H_mul_user, w_near, Nt1, Nt2, N_user, num_gamma, num_theta, num_dis)


    code_reshape = reshape(w_near, Nt1*Nt2,[]);
    beam_near_mul = abs(H_mul_user*conj(code_reshape));
    
    % beam selecting
    precoding_matrix_near_near = zeros(Nt1*Nt2, N_user);
    max_idx_near = zeros(1, N_user);
    
    w_near_list = reshape(w_near,Nt1*Nt2,[]);
    beam_gain = abs(H_mul_user*conj(w_near_list));
    [max_value, max_idx] = max(beam_gain, [], 2);
    precoding_matrix_near_near = conj(w_near_list(:, max_idx));
    
%     [row, col, page] = ind2sub([num_gamma, num_theta, num_dis], max_idx);
%     precoding_matrix_near_near = conj(w_near(:, row, col, page));
    
%     for i_user = 1:N_user
%         [max_value, max_idx] = max(beam_near_mul(i_user,:));
%         max_idx_near(i_user) = max_idx;
%         
%         [row, col, page] = ind2sub([num_gamma, num_theta, num_dis], max_idx);
%         code = w_near(:, row, col, page);
% 
%         precoding_matrix_near_near(:, i_user) = conj(code);
%     end
    
    
end

