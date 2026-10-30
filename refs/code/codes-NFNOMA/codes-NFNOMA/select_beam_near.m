function [precoding_matrix_near_near] = select_beam_near(H_mul_user, w_near, Nt, N_user, num_theta, num_dis)
%SELECT_BEAM_NEAR 此处显示有关此函数的摘要
%   此处显示详细说明

    beam_near_mul2 = zeros(N_user, num_theta*num_dis);
    cnt = 1;
    for i_row = 1:num_theta
        for j_col = 1:num_dis
            beam_near_mul2(:, cnt) = abs(H_mul_user*conj(w_near(:,i_row,j_col)));
            cnt = cnt +1;
        end
    end
    
    % beam selecting
    precoding_matrix_near_near = zeros(Nt, N_user);
    max_idx_near = zeros(1, N_user);
%     beam_select_near = zeros(1,num_theta*num_dis);
%     beam_value_near = zeros(1,num_theta*num_dis);

    for i_user = 1:N_user
        [~, max_idx] = max(beam_near_mul2(i_user,:));
        max_idx_near(i_user) = max_idx;

        idx_a = ceil(max_idx/num_dis);
        idx_b = max_idx - num_dis*(idx_a-1);
        precoding_near = conj(w_near(:,idx_a,idx_b));
        precoding_matrix_near_near(:, i_user) = precoding_near;
    end
    
    
end

