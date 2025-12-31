## Setup guide

This is not a formal setup instruction, but an informal guide to help with the installation of the required packages if importing the requirements.txt file is not working.

We recommend using a version of Python 3.9, it has worked stably for us.

1. Use Python 3.9.X
2. Create venv: python3.9 -m venv venv_X, source venv_X/bin/activate
3. Install tf 2.13.0: pip install tensorflow==2.13.0
4. Uninstall numpy (1.24.X), install numpy 1.23.5: pip uninstall numpy, pip install numpy==1.23.5
5. Install gym: pip install gym
6. Install plt 3.5.3: pip install matplotlib==3.5.3
7. Install 'gym[classic_control]' ('' required): pip install 'gym[classic_control]'
8. Change CartPole-v0 to CartPole-v1 in code