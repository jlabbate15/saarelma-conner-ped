import sys
from pathlib import Path
import numpy as np

ROOT = Path.cwd().parent.parent.parent  # saarelma-conner-ped
sys.path.insert(0, str(ROOT / "dependencies" / "epednn_mit" / "src"))

from epednn_mit.models.sparc.tensorflow_model import generate_epednn_mit_sparc_tensorflow
from epednn_mit.utils.load import load_weights

root = ROOT / "dependencies" / "epednn_mit" / "src" / "epednn_mit" / "models" / "sparc"
weights = load_weights(sorted(root.glob("*sparc*.pkl")))
model = generate_epednn_mit_sparc_tensorflow(weights)

''' Training dataset was on: 
Ip:     [  1.6  , 14.3   ]
Bt:     [  7.2  , 12.2   ]
R:      [  1.85 ,  1.85  ]
a:      [  0.57 ,  0.57  ]
kappa:  [  1.53 ,  2.29  ]
delta:  [  0.39 ,  0.59  ]
neped:  [  2.84 , 90.235 ]
betan:  [  0.8  ,  1.6   ]
zeff:   [  1.3  ,  2.5   ]
'''

Ip = 2.0
Bt = 8.0
R = 1.85
a = 0.57
kappa = 1.70
delta = 0.40
neped = 10.0
betan = 1.0
zeff = 1.5
x = np.atleast_2d([Ip, Bt, R, a, kappa, delta, neped, betan, zeff])
print(model.predict(x))  # [[ped_height, ped_width]]