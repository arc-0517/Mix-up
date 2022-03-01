#%%
import sys
import gc
import time
import itertools
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

hyperparameter_setting = dict(mixup_type=['no_mixup', 'mix_up', 'cut_mix', 'manifold_mixup'])

hyperparameter_setting = itertools.product(*[hyperparameter_setting[i] for i in hyperparameter_setting.keys()])

for param in hyperparameter_setting:
    script_descriptor = open('./main.py', encoding='utf-8')
    a_script = script_descriptor.read()
    sys.argv = ["./main.py",
                '--mixup_type', f'{param[0]}']
    try:
        print(sys.argv)
        print('start')
        exec(a_script)
        gc.collect()
    except:
        print('failed')