clear all; clc;

% rng(0);
f_c = 2.4e9;        % Frequency
Lambda = 3e8/f_c;   % Wavelength

N = 256;

W = 10;

L_num = [22:2:42];

for qq = 1:length(L_num)

L = L_num(qq);

x = linspace(0,W*Lambda,N)';

Kernal_J0 = zeros(N,N);
Kernal_exp = zeros(N,N);
for aa = 1:N
    for bb = 1:N
        Kernal_J0(aa,bb) = besselj(0,2*pi*norm(x(bb)-x(aa))/Lambda);
        Kernal_exp(aa,bb) = exp(-2*pi/Lambda*norm(x(bb)-x(aa))^2/1);
    end
end
[U,S,~] = svd(Kernal_J0);

Rep = 100;
Kernal_SV_mean = zeros(N,N);
Kernal_CDL_mean = zeros(N,N);
for rp = 1:Rep
    h = SV_channel(x,Lambda,L);
    Kernal_SV_mean = Kernal_SV_mean + h*h';
    h = CDL_channel(x, f_c, Lambda);
    Kernal_CDL_mean = Kernal_CDL_mean + h*h';
end
Kernal_SV_mean = Kernal_SV_mean/Rep;
Kernal_CDL_mean = Kernal_CDL_mean/Rep;

save(['Kernel_L/Kernal' num2str(L) '.mat'],'N','x','L','Kernal_J0','U','S','Kernal_exp','Kernal_SV_mean','Kernal_CDL_mean');

end


