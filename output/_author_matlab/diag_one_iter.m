function diag_one_iter()
% 单 SNR、单信道实现的分段计时——定位 main_near_field_NOMA 单迭代耗时。
% 算法/功率/噪声/优化器设置与原文一致：n=512,K=4,N=K/2,P=1,SNR=16dB(卡顿处),
% Imax=30, fmincon options 同 NF_NOMA(LargeScale off, display iter)。
% 仅计时，不改任何官方文件。rng(0) 固定信道便于复现（主程序本身无种子）。
off = 'C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\codes-NFNOMA\codes-NFNOMA';
cd(off); addpath(off, '-begin');
rng(0);

n = 512; d = 0.005; lam = 0.01; fc = 3e8/lam;
K = 4; N = K/2; P = 1; Imax = 30;
SNR_dB = 16; SNR = 10^(SNR_dB/10); sigma2 = 1/SNR;
rho_min = 4; delta = 1.8;
r_min = 4; r_max = 40; sigma_aod = pi/180*5; L = 5; kappa = 8;

fprintf('== 单迭代分段计时: SNR=%ddB, K=%d, n=%d, Imax=%d, rng(0) ==\n', SNR_dB, K, n, Imax);

t = tic; [w, ~, ~] = QuaCode(n, d, lam, delta, rho_min); t_q = toc(t);
wr = reshape(w, n, []); nt = size(w, 2); nd = size(w, 3);
fprintf('  QuaCode            : %8.3f s  (码本 %dx%d=%d)\n', t_q, nt, nd, nt*nd);

t = tic; H = generate_user_in_circle_multipath(n, d, r_min, r_max, fc, K, sigma_aod, L, kappa); t_ch = toc(t);
fprintf('  信道生成            : %8.3f s\n', t_ch);

t = tic; pm = select_beam_near(H, w, n, K, nt, nd); t_sb = toc(t);
fprintf('  select_beam_near   : %8.3f s  ★main:129\n', t_sb);

% LDMA 支路（main:131-163，含噪声构造）
t = tic;
noise_sigma = 1/SNR;
noise = sqrt(noise_sigma/n)*(randn(K,n)+1i*randn(K,n))/sqrt(2);
He = (pm.'*(H+noise).').';
Fd = cal_zero_forcing(He, pm);
sr = cal_sum_rate_mu(H, pm*Fd, K, sigma2);
t_ldma = toc(t);
fprintf('  LDMA 支路(含噪声/ZF): %8.3f s  (rate=%.4f)\n', t_ldma, sr);

% NOMA 分组链（main:168-171）
t = tic; Ht = (H*conj(wr)).'; [Hr, F, rf, setf] = NF_group_1(Ht, K, Ht); t_g1 = toc(t);
fprintf('  NF_group_1 链       : %8.3f s  (rf=%d)\n', t_g1, rf);

% NF_group_4（main 调用但输出未被使用）
t = tic; [Hr4, F4, setf4] = NF_group_4(Ht, K, N, n, d, r_min, r_max, fc, sigma_aod, L, kappa, wr); t_g4 = toc(t);
fprintf('  NF_group_4(死代码)  : %8.3f s ★main 调用但未使用其输出\n', t_g4);

% NF_NOMA：Imax=1 / 2 / 30 → 外层单步边际成本
t = tic; [SE1, ~, ~, pw1] = NF_NOMA(Hr, F, setf, K, rf, sigma2, P, 1);  t_n1  = toc(t);
t = tic; [SE2, ~, ~, pw2] = NF_NOMA(Hr, F, setf, K, rf, sigma2, P, 2);  t_n2  = toc(t);
t = tic; [SE30, ~, ~, pw30] = NF_NOMA(Hr, F, setf, K, rf, sigma2, P, Imax); t_n30 = toc(t);
fprintf('  NF_NOMA Imax=1     : %8.3f s  SE(end)=%.4f\n', t_n1, SE1(end));
fprintf('  NF_NOMA Imax=2     : %8.3f s\n', t_n2);
fprintf('  NF_NOMA Imax=30    : %8.3f s ★main:190  SE(end)=%.4f sum_p=%.4f\n', t_n30, SE30(end), sum(pw30));
fprintf('  外层单步边际(30-1)/29: %7.3f s\n', (t_n30-t_n1)/29);

% 单次 fmincon（与 NF_NOMA 内同款约束与 options）+ 单次 myfun_NF
Fi = F;
for k = 1:K
    Fi(:,k) = Fi(:,k)/norm(Fi(:,k));
end
H_eq = Hr' * Fi;
% 造 A,C：与 NF_NOMA 内层同公式（用 power0）
power0 = zeros(rf, K);
for nn = 1:rf
    usern = setf(nn,:); usern(usern==0) = [];
    for m = 1:length(usern)
        power0(nn,m) = m*(norm(Fi(:,usern(m))))^2;
    end
end
power0 = power0*(P/sum(power0(:)));
p0v = [];
ind = zeros(1,rf); usel = 0;
for nn = 1:rf
    usern = setf(nn,:); usern(usern==0) = [];
    ind(nn) = usel+length(usern); usel = ind(nn);
end
ind = [0, ind];
A = zeros(rf,K); C = zeros(rf,K); E = zeros(rf,K);
for nn = 1:rf
    usern = setf(nn,:); usern(usern==0) = [];
    for m = 1:length(usern)
        bet = abs(H_eq(usern(m),usern(m)))^2*sum(power0(nn,1:m-1)) + sigma2;
        for j = 1:rf
            if j ~= nn
                bet = bet + abs(H_eq(usern(m),setf(j,1)))^2*sum(power0(j,:));
            end
        end
        C(nn,m) = (sqrt(power0(nn,m))*H_eq(usern(m),usern(m)))'/(power0(nn,m)*abs(H_eq(usern(m),usern(m)))^2+bet);
        E(nn,m) = 1 - power0(nn,m)*abs(H_eq(usern(m),usern(m)))^2/(power0(nn,m)*abs(H_eq(usern(m),usern(m)))^2+bet);
        A(nn,m) = 1/E(nn,m);
    end
end
for nn = 1:rf
    usern = setf(nn,:); usern(usern==0) = [];
    for m = 1:length(usern)
        p0v = [p0v; power0(nn,m)];
    end
end
% 约束块：NF_NOMA:66-101 同款
A0 = ones(1,K); b0 = P;
ita = 2^0.01-1;
indc = zeros(1,rf); usel = 0;
for nn = 1:rf
    usern = setf(nn,:); usern(usern==0) = [];
    indc(nn) = usel+length(usern); usel = indc(nn);
end
indc = [0, indc];
for nn = 1:rf
    usern = setf(nn,:); usern(usern==0) = [];
    for m = 1:length(usern)
        a0 = zeros(1,K);
        a0(indc(nn)+m) = abs(H_eq(usern(m),usern(m)))^2;
        a0(indc(nn)+1:indc(nn)+m-1) = -ita*abs(H_eq(usern(m),usern(m)))^2;
        for j = 1:rf
            if j ~= nn
                usern_j = setf(j,:); usern_j(usern_j==0) = [];
                a0(indc(j)+1:indc(j+1)) = -ita*abs(H_eq(usern(m),usern_j(1)))^2;
            end
        end
        A0 = [A0; -a0];
        b0 = [b0; -ita*sigma2];
    end
end
lb = zeros(K,1); ub = zeros(K,1); ub(1:K) = P;
options = optimset('LargeScale','off','display','iter');
t = tic; [pp, sume] = fmincon(@(pw) myfun_NF(pw,Hr,Fi,A,C,setf,K,rf,sigma2), p0v, A0, b0, [], [], lb, ub, [], options); t_fm = toc(t);
fprintf('  单次 fmincon(同款)  : %8.3f s  (K=%d 个功率变量)\n', t_fm, K);

t = tic; for i = 1:100, myfun_NF(p0v,Hr,Fi,A,C,setf,K,rf,sigma2); end; t_mf = toc(t)/100;
fprintf('  单次 myfun_NF      : %8.3e s\n', t_mf);

fprintf('\n== 合计（main 单迭代 = 信道+选波+分组+NOMA） ==\n');
fprintf('  选波+分组+NOMA(30) = %.2f s   其中 NF_group_4=%.2f s, select_beam_near=%.2f s\n', ...
    t_sb+t_g1+t_g4+t_n30, t_g4, t_sb);
end
