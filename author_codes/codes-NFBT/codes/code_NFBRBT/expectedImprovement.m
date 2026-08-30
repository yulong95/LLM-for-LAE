function [ei, maxEIIndex] = expectedImprovement(mu, sigma, f_x_plus, xi)
    % 计算改进的 Z 分数
    Z = (mu - f_x_plus - xi) ./ sigma;
    
    % 初始化 EI 向量
    ei = zeros(size(mu));
    
    % 确保标准差为正
    valid = sigma > 0;
    
    % 计算有效的 EI 值
    ei(valid) = (mu(valid) - f_x_plus - xi) .* normcdf(Z(valid)) + ...
                    sigma(valid) .* normpdf(Z(valid));
    
    % 找到最大 EI 值的索引
    [maxEI, maxEIIndex] = max(ei);
    
    % 打印最大的 EI 值和对应的索引
    fprintf('最大 EI 值: %f，在索引: %d\n', maxEI, maxEIIndex);
end
