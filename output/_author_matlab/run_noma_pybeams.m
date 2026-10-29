function run_noma_pybeams(input_file, output_file)
% 在 Python 现行评估的波束/用户顺序上原样运行作者 NF_NOMA（不改算法，仅接口适配）。
% 输入:  H_batch [B,K,N] complex, Fi_batch [B,N,K] complex(=conj(u_k)),
%        setf_batch [B,K,K] 1-based(0 补位), rf_nums [B], sigma2, P_total, Imax
% 输出:  power_user [B,K]（fmincon 最终功率，按用户索引）, SE_end [B], fail, elapsed
data = load(input_file);
H_batch = data.H_batch;
Fi_batch = data.Fi_batch;
setf_batch = data.setf_batch;
rf_nums = data.rf_nums(:);
sigma2 = data.sigma2;
P_total = data.P_total;
Imax = data.Imax;
[B, K, ~] = size(H_batch);

power_user = nan(B, K);
SE_end = nan(B, 1);
fail = 0;

tic;
for b = 1:B
    H_b = squeeze(H_batch(b, :, :));        % [K,N]
    Fi = squeeze(Fi_batch(b, :, :));        % [N,K]
    rf = rf_nums(b);
    setf = squeeze(setf_batch(b, 1:rf, :)); % [rf,K] 1-based
    try
        [SE, ~, ~, pw] = NF_NOMA(H_b.', Fi, setf, K, rf, sigma2, P_total, Imax);
        SE_end(b) = SE(end);
        ind = zeros(1, rf); usel = 0;
        for n = 1:rf
            usern = setf(n, :); usern(usern == 0) = [];
            ind(n) = usel + length(usern); usel = ind(n);
        end
        ind = [0, ind];
        for n = 1:rf
            usern = setf(n, :); usern(usern == 0) = [];
            for m = 1:length(usern)
                power_user(b, usern(m)) = pw(ind(n) + m);
            end
        end
    catch
        fail = fail + 1;
    end
    if mod(b, 50) == 0
        fprintf('  %d/%d  SE_end=%.4f  sum_p=%.4f\n', b, B, SE_end(b), nansum(power_user(b, :)));
    end
end
elapsed = toc;

fprintf('\n===== PYBEAMS SUMMARY =====\n');
fprintf('samples=%d  fail=%d  elapsed=%.1f s (%.3f s/sample)\n', B, fail, elapsed, elapsed / B);
fprintf('SE_end mean=%.4f（旧 A,C 下界，仅参考）\n', mean(SE_end, 'omitnan'));
fprintf('sum_p mean=%.4f min=%.4f max=%.4f（P=1）\n', ...
    mean(sum(power_user, 2, 'omitnan')), ...
    min(sum(power_user, 2, 'omitnan')), max(sum(power_user, 2, 'omitnan')));
save(output_file, 'power_user', 'SE_end', 'fail', 'elapsed');
fprintf('Saved: %s\n', output_file);
end
