import json
new = json.load(open(r'output\eval_baselines_results.json'))
old = json.load(open(r'output\eval_baselines_results.python_noma.bak.json'))
paper_k = {'5': 18.0333, '6': 20.1004, '7': 21.8992, '8': 23.7766, '9': 25.2912, '10': 26.6839}
paper_p = {'-10': 6.8764, '-8': 9.5466, '-6': 12.8684, '-4': 16.9356, '-2': 21.5639,
           '0': 26.7373, '2': 32.4356, '4': 38.1334, '6': 44.2091, '8': 50.2751, '10': 56.4159}
print('== Fig.6 K (NF-NOMA) ==')
print(f'{"K":>3} {"paper":>8} {"official":>9} {"python_old":>10} {"new-paper":>9} {"new-old":>8}')
for k in ['5', '6', '7', '8', '9', '10']:
    n = new['k_sweep'][k]['NF-NOMA']; o = old['k_sweep'][k]['NF-NOMA']; p = paper_k[k]
    print(f'{k:>3} {p:8.4f} {n:9.4f} {o:10.4f} {n-p:+9.4f} {n-o:+8.4f}')
print('== Fig.9 P (NF-NOMA) ==')
print(f'{"P":>4} {"paper":>8} {"official":>9} {"python_old":>10} {"new-paper":>9} {"new-old":>8}')
for p in ['-10', '-8', '-6', '-4', '-2', '0', '2', '4', '6', '8', '10']:
    n = new['fig9'][p]['NF-NOMA']; o = old['fig9'][p]['NF-NOMA']; pp = paper_p[p]
    print(f'{p:>4} {pp:8.4f} {n:9.4f} {o:10.4f} {n-pp:+9.4f} {n-o:+8.4f}')
print('== 其余 baseline 未变性检查 ==')
for k in ['5', '10']:
    for m in ['Capacity', 'LDMA', 'SDMA']:
        same = abs(new['k_sweep'][k][m] - old['k_sweep'][k][m]) < 1e-9
        print(f'  K={k} {m}: {new["k_sweep"][k][m]:.4f} unchanged={same}')
for p in ['-10', '0', '10']:
    for m in ['Capacity', 'LDMA', 'SDMA']:
        same = abs(new['fig9'][p][m] - old['fig9'][p][m]) < 1e-9
        print(f'  P={p} {m}: {new["fig9"][p][m]:.4f} unchanged={same}')
print('== Fig.7/8 NOMA 常数线 ==')
print(f'  fig7 alpha=0.4 NF-NOMA = {new["fig7"]["0.4"]["NF-NOMA"]:.4f}（应=24.903 水平线）')
print(f'  fig8 rmin=0.0..1.0 NF-NOMA = {[round(new["fig8"][r]["NF-NOMA"],4) for r in ["0","0.2","0.4","0.6","0.8","1.0"]]}')
