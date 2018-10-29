import os,\
    sys,\
    inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from generator.generator import generator
import numpy as np
import matplotlib.pyplot as plt


def relations(lOr):

    ge = generator(64, root_path='..')
    lOi = [ge.listOFdataname.index(i) for i in lOr]
    print(lOi)

    data, target = next(ge)
    target, data = zip(*sorted(zip(target, data)))

    x = np.linspace(0, 1, 64)

    plt.plot(x, target, label='target')
    for ind, i in enumerate(lOi):
        di = np.array(data)[:, i]
        di = di/np.amax(di)
        plt.plot(x, di, label=lOr[ind])

    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.show()


if __name__ == '__main__':

    lOr = ['heals',
           'damageDealt',
           'boosts',
           'walkDistance']
    relations(lOr)
