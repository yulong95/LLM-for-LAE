function [h_hat] = SeLMMSE(x,h,sigma2,P)

N = length(x);
h_hat = zeros(N,1);

Sample = floor(linspace(1,N,P));
h_hat(Sample) = h(Sample) + sqrt(sigma2)*(randn(P,1)+1j*randn(P,1))/sqrt(2);

for n = 1:N    
    [~,idx] =min(abs(n-Sample));
    h_hat(n) = h_hat(Sample(idx));
end

end

