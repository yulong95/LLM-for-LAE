import sys, os, json, importlib.util
sys.path.insert(0, r'C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1')
spec = importlib.util.spec_from_file_location(
    'e2b', r'C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1\output\mini_exp_E2b_beams.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

results = []
results.append(m.train_one('wmed', m.CNN_Wmed(M=16), epochs=25))
results.append(m.train_one('wres', m.CNN_Wres(), epochs=25))
out = os.path.join(m.base_output, 'mini_exp_E2b_wmed_wres.json')
json.dump(results, open(out, 'w'), indent=2)
print('SUMMARY')
for r in results:
    print(f"{r['name']:10s} test={r['test_rate']:.4f} acc={r['test_acc']:.3f} gap={r['gap']:+.4f}")
print('saved', out)
