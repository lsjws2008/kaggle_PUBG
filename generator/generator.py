import os,\
    sys,\
    inspect
import numpy as np
from random import shuffle
from pympler import asizeof

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from tools.normalizers import normalizer
from global_config.global_config import config


class generator():
    def __init__(self,
                 batch,
                 csv_name='train.csv',
                 root_path='.'):
        """
        Id,groupId,matchId,assists,boosts,damageDealt,DBNOs,headshotKills,heals,
        killPlace,killPoints,kills,killStreaks,longestKill,maxPlace,numGroups,
        revives,rideDistance,roadKills,swimDistance,teamKills,vehicleDestroys,
        walkDistance,weaponsAcquired,winPoints,winPlacePerc

        """
        self.train_data = []
        self.IDs = []
        self.types = csv_name.split('.')[0]
        p = os.path.abspath(root_path)
        trainfile = os.path.join(p,
                                 'datas',
                                 csv_name)

        with open(trainfile) as f:
            lines = f.readlines()
            self.listOFdataname = lines[0][:-2].split(',')[3:]
            for i in lines[1:]:
                line = i[:-2].split(',')
                self.IDs.append(line[0])
                line = line[3:]
                line = [float(j) for j in line]
                self.train_data.append(line)
        self.train_data = normalizer(self.train_data)

        if self.types == 'train':
            np.random.shuffle(self.train_data)
        self.batch = batch
        self.cfg = config()
        self.fuc = self.gen_fuc()

    def generator_train_group(self, loi):
        if len(self.train_data) < self.batch*2:
            self.__init__(self.batch)
        datas = []
        labels = []
        for i in range(self.batch):
            datas.append(self.train_data[0][:-1])
            labels.append(self.train_data[0][-1])
            self.train_data = self.train_data[1:]

        datas = np.expand_dims(np.array(datas)[:, loi],
                               -1)
        return datas, labels

    def generator_test_group(self, loi):
        datas = []
        ID = []
        for i in range(self.batch):
            if len(self.IDs) == 0:
                break
            datas.append(self.train_data[0][:-1])
            ID.append(self.IDs[0])
            self.train_data = self.train_data[1:]
            self.IDs.remove(self.IDs[0])
        datas = np.expand_dims(np.array(datas)[:, loi],
                               -1)
        return datas, ID

    def gen_fuc(self):
        if self.types == 'test':
            return self.generator_test_group
        if self.types == 'train':
            return self.generator_train_group

    def __next__(self, lor=None):

        if lor is None:
            lor = self.cfg.lOr
            if self.types == 'test':
                lor = lor[:-1]
        loi = [self.listOFdataname.index(i) for i in lor]

        return self.fuc(loi)


if __name__ == '__main__':
    ge = generator(1000,
                   root_path='..')
    print(np.mean(next(ge)[1]))
