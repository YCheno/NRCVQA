import os
import skvideo.io
from sklearn.model_selection import KFold
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json


if __name__ == "__main__":
    matPath = '/mnt/wwn-0x5000cca0c3e1998a/finetune_database/Real_cartoon+_info/mos/Real_cartoon.mat'
    filePrefix = '/mnt/wwn-0x5000cca0c3e1998a/finetune_database/Real_cartoon+_info/Real_cartoon+/'

    
    data = sio.loadmat(matPath)
    name, mos = data['video_names'], data['scores']
    #refIdx, refName = data['ref_index'], data['ref_name']
    length = len(name)

    kf = KFold(n_splits=5, shuffle=True)

    for i, (trainIdx, testIdx) in enumerate(kf.split(range(length))):
        ret = OrderedDict()
        ret['train'] = OrderedDict()
        ret['test'] = OrderedDict()

        trn_dis = []
        trn_mos = []
        trn_height = []
        trn_width = []


        tst_dis = []
        tst_mos = []
        tst_height = []
        tst_width = []

        for idx in trainIdx:
            suffix = name[idx][0][0]
            videoName = os.path.join(filePrefix, suffix)
            videoMos = mos[idx][0]
            video_data = skvideo.io.vread(videoName)
            #print(video_data.shape)
          #  videoType = disType[idx][0]

            trn_dis.append(suffix)
            trn_mos.append(float(videoMos))
           # trn_type.append(int(videoType))
            trn_height.append(video_data.shape[1])
            trn_width.append(video_data.shape[2])
           # trn_fps.append(0)

        for idx in testIdx:
            suffix = name[idx][0][0]
            videoName = os.path.join(filePrefix, suffix)
            videoMos = mos[idx][0]
            video_data = skvideo.io.vread(videoName)
          #  videoType = disType[idx][0]

            tst_dis.append(suffix)
            tst_mos.append(float(videoMos))
           # tst_type.append(int(videoType))
            tst_height.append(video_data.shape[1])
            tst_width.append(video_data.shape[2])
           # tst_fps.append(0)

        print("train num : ", len(trn_dis))
        print("test num : ", len(tst_dis))
        ret['train']['dis'] = trn_dis
        # ret['train']['ref'] = trn_ref
        ret['train']['mos'] = trn_mos
      #  ret['train']['type'] = trn_type
        ret['train']['height'] = trn_height
        ret['train']['width'] = trn_width
       # ret['train']['fps'] = trn_fps

        ret['test']['dis'] = tst_dis
        # ret['test']['ref'] = tst_ref
        ret['test']['mos'] = tst_mos
      #  ret['test']['type'] = tst_type
        ret['test']['height'] = tst_height
        ret['test']['width'] = tst_width
      #  ret['test']['fps'] = tst_fps

       # i =i + 20
        res_path = f"./Cartoon_subj_score_{i}.json"
        with open(res_path, 'w') as f:
            json.dump(ret, f, indent=4)
