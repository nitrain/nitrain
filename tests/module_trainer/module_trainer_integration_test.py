

import os
os.chdir('/users/ncullen/desktop/projects/torchsample/examples/core')

for script in os.listdir('.'):
    exec(script+'.py')