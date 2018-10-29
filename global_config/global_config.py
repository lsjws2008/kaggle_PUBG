import tensorflow as tf

class config():
    def __init__(self):
        self.weight_decay = tf.constant(0.0005, dtype=tf.float32)
        self.model = 'vgg19'
        # 'default'
        # self.layer = [128,
        #               1024,
        #               1024,
        #               1024,
        #               1024,
        #               1024,
        #               1024,
        #               128,
        #               1]
        self.layer = [['conv1d', 3, 1, 64],
                      ['conv1d', 3, 64, 64],
                      ['maxpooling', 2],
                      ['conv1d', 3, 64, 128],
                      ['conv1d', 3, 128, 128],
                      ['maxpooling', 2],
                      ['conv1d', 3, 128, 256],
                      ['conv1d', 3, 256, 256],
                      ['conv1d', 3, 256, 256],
                      ['conv1d', 3, 256, 256],
                      ['maxpooling', 2],
                      ['conv1d', 3, 256, 512],
                      ['conv1d', 3, 512, 512],
                      ['conv1d', 3, 512, 512],
                      ['conv1d', 3, 512, 512],
                      ['maxpooling', 2],
                      ['conv1d', 3, 512, 512],
                      ['conv1d', 3, 512, 512],
                      ['conv1d', 3, 512, 512],
                      ['conv1d', 3, 512, 512],
                      ['maxpooling', 2],
                      ['flatten', 512],
                      ['fc', 512, 4096],
                      ['fc', 4096, 1000],
                      ['fc', 1000, 1]]
        print('layers number:', len(self.layer))
        self.batch = 64
        self.epoch = 1000000
        self.target_shape = 1
        self.train_csv = 'train.csv'
        self.test_csv = 'test.csv'
        self.learning_rate = 1e-3
        self.momentum = 0.9
        self.dropout = 0.8
        # self.lOr = ['heals',
        #             'damageDealt',
        #             'boosts',
        #             'walkDistance']
        self.delOr = 'swimDistance,'
        self.lOr = 'Id,groupId,matchId,assists,boosts,damageDealt,DBNOs,headshotKills,heals,killPlace,killPoints,kills,killStreaks,longestKill,maxPlace,numGroups,revives,rideDistance,roadKills,teamKills,vehicleDestroys,walkDistance,weaponsAcquired,winPoints,winPlacePerc'
        self.lOr = self.lOr.split(',')[3:][:-1]
        self.input_shape = len(self.lOr)
        self.decay_epoch = 2e4
        self.decay_rate = 3
