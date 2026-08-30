function negLogLikelihood = gprObjectiveFunction(params, X, y)
    sigma_f = params(1);
    l = params(2);
    % 计算指数核函数矩阵
    N = size(X, 1);
    K = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K(i, j) = sigma_f^2 * exp(-norm(X(i,:)-X(j,:))^2 / (2*l^2));
        end
    end
    % 添加噪声项
    sigma_n = 0.1; % 假定噪声方差，根据实际情况调整
    K = K + sigma_n^2 * eye(N);
    % 计算负对数似然
    % 使用 Cholesky 分解来避免直接计算逆和行列式
    L = chol(K, 'lower');
    alpha = L'\(L\y);
    negLogLikelihood = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*N*log(2*pi);
   
end