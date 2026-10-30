function [H, c, lambda] = generate_ULA_rotate(fc, Nt, Nr,D_list, theta)
%GENERATE_ULA_PARALLEL 此处显示有关此函数的摘要
%   此处显示详细说明
c = 3e8;
lambda = c/fc;

[~, d_len] = size(D_list);

tx_start = -(Nt/2-0.5)*lambda/2;
tx_loc = tx_start:lambda/2:-tx_start;

rx_start = -(Nr/2-0.5)*lambda/2;
rx_start_x = rx_start*sin(theta);
rx_start_y = rx_start*cos(theta);
rx_loc_x = linspace(rx_start_x, -rx_start_x, Nr);
rx_loc_y = linspace(rx_start_y, -rx_start_y, Nr);

H = zeros(d_len, Nr, Nt);
for i_d = 1:d_len
    D = D_list(i_d);
    dis_matrix = zeros(Nt, Nr);
    for i_tx = 1:Nt
        for i_rx = 1:Nr
            dis_matrix(i_tx, i_rx) = sqrt((D-rx_loc_x(i_rx))^2 + (tx_loc(i_tx) - rx_loc_y(i_rx))^2);
        end
    end
%     H_tmp = (exp(-1i*2*pi/lambda*dis_matrix)./dis_matrix).';
    H_tmp = (exp(-1i*2*pi/lambda*dis_matrix)).';
    H_tmp = H_tmp/sqrt(Nt*Nr);
%     H(i_d, :, :) = H_tmp/sqrt(Nt*Nr);
H(i_d, :, :) = H_tmp;
end

