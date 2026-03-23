import os
import skvideo.io
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json


if __name__ == "__main__":
    repeat = 5
    need = 2

    all_scenes = ['0425_001', '0425_011', '0426_001', '0426_012', '0427_001',
                  '0427_012', '0431_001', '0431_011', '0433_002', '0433_013',
                  '0434_001','0434_013']

    matPath = '/mnt/wwn-0x5000cca0c3e1998a/finetune_database/Syn_cartoon+_info/mos/Syn_cartoon0.mat'
    filePrefix = '/home1/server823-2/database/2D-Video/CSIQVideo/'

    
    data = sio.loadmat(matPath)
    name, mos, disType = data['video_names'], data['scores'], data['dis_type']

    for i in range(repeat):
        length = len(all_scenes)
        randomIdx = np.random.permutation(np.arange(length))
        print(randomIdx)
        test_scenes = set()
        for idx in randomIdx[:need]:
            test_scenes.add(all_scenes[idx])

        ret = OrderedDict()
        ret['train'] = OrderedDict()
        ret['test'] = OrderedDict()

        trn_dis = []
        trn_mos = []
        trn_height = []
        trn_width = []
        trn_fps = []
        trn_type = []

        tst_dis = []
        tst_mos = []
        tst_height = []
        tst_width = []
        tst_fps = []
        tst_type = []

        for idx in range(len(name)):
            curScen = '_'.join(name[idx][0][0].split('_')[:2])
            # fps = name[idx][0][0].split('_')[1][:2]
            fps = 0
            # Name DMOS DisType
            suffix = name[idx][0][0]
            videoMos = mos[idx][0]
            videoType = disType[idx][0]

            if curScen in test_scenes:
                tst_dis.append(suffix)
                tst_mos.append(float(videoMos))
                tst_type.append(int(videoType))
                tst_height.append(1280)
                tst_width.append(720)
                tst_fps.append(float(fps))
            else:
                trn_dis.append(suffix)
                trn_mos.append(float(videoMos))
                trn_type.append(int(videoType))
                trn_height.append(1280)
                trn_width.append(720)
                trn_fps.append(float(fps))

        print("train num : ", len(trn_dis))
        print("test num : ", len(tst_dis))
        ret['train']['dis'] = trn_dis
        # ret['train']['ref'] = trn_ref
        ret['train']['mos'] = trn_mos
        ret['train']['type'] = trn_type
        ret['train']['height'] = trn_height
        ret['train']['width'] = trn_width
        ret['train']['fps'] = trn_fps

        ret['test']['dis'] = tst_dis
        # ret['test']['ref'] = tst_ref
        ret['test']['mos'] = tst_mos
        ret['test']['type'] = tst_type
        ret['test']['height'] = tst_height
        ret['test']['width'] = tst_width
        ret['test']['fps'] = tst_fps

        path = './Cartoon_LEAVE2_' + str(i) + '.json'
        with open(path, 'w') as f:
            json.dump(ret, f, indent=4)
