# scripts/setup_libs.py

import os
import subprocess

LIBS_DIR = os.path.join(os.path.dirname(__file__), '..', 'libs')
REPOS = {
    'MachineScientist': 'https://bitbucket.org/rguimera/machine-scientist.git',
    'DeepSymbolicOptimisation': 'git@github.com:jamesbut/deep-symbolic-optimization.git',
    'SRBench': 'git@github.com:cavalab/srbench.git',
    'pmlb': 'git@github.com:EpistasisLab/pmlb.git'
}


def clone_or_pull(name, url):
    path = os.path.join(LIBS_DIR, name)
    if not os.path.exists(path):
        print(f"Cloning {name}...")
        subprocess.check_call(['git', 'clone', url, path])
    else:
        print(f"Updating {name}...")
        subprocess.check_call(['git', '-C', path, 'pull'])


os.makedirs(LIBS_DIR, exist_ok=True)

for name, url in REPOS.items():
    clone_or_pull(name, url)
