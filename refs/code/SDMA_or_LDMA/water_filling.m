function palloc_matrix = water_filling(sigma, P, max_size)

% loss, Kx1向量，即L
% P, 要分配的总功率
% A, 对角阵，即权重alpha
% palloc_matrix, 对角阵，功率分配结果，迹等于P

[height, len] = size(sigma);
Nr = min(height, len);
loss = zeros(1,Nr);
for i = 1:Nr
    loss(i) = 1/(sigma(i,i)^2);
end

K = length(loss); % 用户数

w = ones(1,Nr); % width, 台阶宽度
h = loss./w; % height, 台阶高度

allo_set = 1:K; % 初始化要注水用户的索引集合
level = (P+sum(loss(allo_set>0)))/sum(w(allo_set>0)); % “虚拟”水位线
[h_hat, k_hat] = max(fliplr(h(allo_set>0)));
k_hat = length(h(allo_set>0)) - k_hat + 1;

while h_hat>=level

    allo_set(k_hat) = -1;
    level = (P+sum(loss(allo_set>0)))/sum(w(allo_set>0)); % “虚拟”水位线
    [h_hat, k_hat] = max(fliplr(h(allo_set>0)));
    k_hat = length(h(allo_set>0)) - k_hat + 1;

end

palloc_matrix = zeros(K);
for k = 1:K
    if allo_set(k)>0
        palloc_matrix(k, k) = (level - h(k))*w(k);
    end
end

% palloc_matrix(max_size, max_size) = 0;
end

