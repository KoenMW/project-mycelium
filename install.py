import subprocess
import sys

# List of packages to install
packages = [
    'numpy', 'opencv-python', 'keras', 'tensorflow', 'scikit-learn', 'matplotlib', 'yellowbrick'
]

# Install packages
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
