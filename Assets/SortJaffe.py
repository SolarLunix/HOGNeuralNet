import os
import shutil
import numpy as np

dir_path = os.path.curdir
folders = np.array([os.path.join(dir_path, "jaffe\\Angry"),
                    os.path.join(dir_path, "jaffe\\Disgust"),
                    os.path.join(dir_path, "jaffe\\Fear"),
                    os.path.join(dir_path, "jaffe\\Happy"),
                    os.path.join(dir_path, "jaffe\\Neutral"),
                    os.path.join(dir_path, "jaffe\\Sad"),
                    os.path.join(dir_path, "jaffe\\Surprise")])

for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)

db = os.path.join(os.path.curdir, 'jaffe')
for root, dirs, filenames in os.walk(db):
    for f in filenames:
        loc = os.path.join(root, f)
        n_loc = root
        if f.__contains__('AN'):
            n_loc = folders[0]
        elif f.__contains__('DI'):
            n_loc = folders[1]
        elif f.__contains__('FE'):
            n_loc = folders[2]
        elif f.__contains__('HA'):
            n_loc = folders[3]
        elif f.__contains__('NE'):
            n_loc = folders[4]
        elif f.__contains__('SA'):
            n_loc = folders[5]
        elif f.__contains__('SU'):
            n_loc = folders[6]

        dest = os.path.join(n_loc, f)
        if loc != dest:
            shutil.move(loc, n_loc)