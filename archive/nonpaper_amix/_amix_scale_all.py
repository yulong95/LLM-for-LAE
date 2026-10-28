"""Sequential A-mix scale-up: TF -> CNN -> GPT2, 100 epochs each."""
import subprocess, sys, os
PY = r"C:\Users\17859\Desktop\files\Grad_Project\venv\Scripts\python.exe"
WD = r"C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1"
log = open(os.path.join(WD, 'output', 'amix_scale_train.log'), 'w')
err = open(os.path.join(WD, 'output', 'amix_scale_train.err'), 'w')
for model in ['tf', 'cnn', 'gpt2']:
    print(f'=== train {model} ===', flush=True)
    r = subprocess.run([PY, 'train_amix.py', '--model', model, '--epochs', '100'], cwd=WD, stdout=log, stderr=err)
    print(f'=== {model} exit={r.returncode} ===', flush=True)
    log.flush()
    if r.returncode != 0:
        sys.exit(r.returncode)
print('ALL TRAIN DONE', flush=True)
