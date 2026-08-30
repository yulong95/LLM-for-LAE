function h = SV_channel(x,Lambda,L)

N = length(x);

h = zeros(N,1);
for l = 1: L
    theta_n = 2*pi/Lambda*(x(2)-x(1))*(2*rand()-1);
    %    theta_n = pi*(2*floor(N*rand())-N+1)/N;
    h = h + 1/sqrt(2)*(randn()+1j*randn())*exp(1j*theta_n*[0:1:N-1]');
end
% h = h + exp(1j*2*pi*rand())* exp(1j*theta_n*[0:1:N-1]');

h = h/sqrt(L);

% h = h/norm(h);
end

