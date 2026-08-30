function [H]=farfield_channel(n,K,L,lamada,d)
% n: number of transmit beams (transmit antennas)
% K: number of users
% L: number of paths, L=1 for LoS, L>1 for multipath
H = zeros(n,K);
theta=zeros(L,K);
for k=1:K
    beta=zeros(1,L); 
    %%% complex gain
%     beta(1:L) = sqrt(2)*(randn(1,L)+1i*randn(1,L));
    beta(1) = 1; % gain of the LoS
%     beta(2:L) = sqrt(0.1*beta(1));
    beta(2:L) = sqrt(0.1*beta(1))*exp(-1i*2*pi*rand(1,L-1)); % gain of
%     NLoS
    %%% DoA

    theta(1,k) = (pi)*rand(1) - pi/2;
    % theta(1,k) = (2*pi/3)*rand(1) - pi/2;
    % theta(1,k) = (pi/3)*rand(1) - pi/6;
    theta(2:L,k) = pi*rand(1,L-1) - pi/2;
    for j = 1:L
        H(:,k) = H(:,k) + beta(j)*array_respones(theta(j,k),n,d,lamada);
    end
end