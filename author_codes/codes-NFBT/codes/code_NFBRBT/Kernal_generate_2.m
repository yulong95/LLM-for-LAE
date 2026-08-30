clear all; clc;

% rng(0);
f_c = 2.4e9;        % Frequency
Lambda = 3e8/f_c;   % Wavelength
L = 6;              % Path number

N = 256;

W_num = [1:1:20];       % Aperture length per wavelength
% W_num(1) = 10;

for qq = 1:length(W_num)

W = W_num(qq);

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

Rep = 1000;
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

save(['Kernel/Kernal' num2str(W) '.mat'],'N','x','L','Kernal_J0','U','S','Kernal_exp','Kernal_SV_mean','Kernal_CDL_mean');

end


