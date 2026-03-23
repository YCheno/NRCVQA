import os
import json
import numpy as np
from collections import OrderedDict, defaultdict


yuv_dir = '/home/amd7302/YXX/Code/Unsupervised/Syn_Database/ref_3000/'

def path_change(file, type):
    suffix = file.split('/')[-1]
    #suffix = suffix.replace('.mp4', '.yuv')
    return suffix


def json_dump(subj_score_file, trainRatio=0.8):

    ALL = []
    ref = os.listdir(yuv_dir)
    for file in ref:
        ALL.append(file)

    randomIdx = np.random.permutation(np.arange(len(ALL)))
    print(len(ALL))
    split = int(np.floor(0.8 * len(ALL)))
    train, test = set(), set()

    for idx in randomIdx[:split]:
        train.add(ALL[idx])
    for idx in randomIdx[split:]:
        test.add(ALL[idx])

    with open(subj_score_file, "r") as f:
        info = json.load(f)


    data = OrderedDict()
    data['test'] = OrderedDict()
    data['train'] = OrderedDict()

    test_ref = []
    test_height = []
    test_width = []

    train_ref = []
    train_height = []
    train_width = []

    # length = len(info['dis'])
    # print(length)
    # randomIdx = np.random.permutation(np.arange(length))
    # split = int(np.floor(0.8 * length))
    # train, test = randomIdx[:split], randomIdx[split:]
    for idx in range(len(info['ref'])):
        suffix = info['ref'][idx].split('/')[-1]
        if suffix in train:
            train_ref.append(info['ref'][idx])
            train_width.append(info['width'][idx])
            train_height.append(info['height'][idx])
        else:
            test_ref.append(info['ref'][idx])
            test_width.append(info['width'][idx])
            test_height.append(info['height'][idx])


    for idx, file in enumerate(train_ref):
        train_ref[idx] = path_change(file, 'ref')

    for idx, file in enumerate(test_ref):
        test_ref[idx] = path_change(file, 'ref')

    data['train']['ref'] = train_ref
    data['train']['width'] = train_width
    data['train']['height'] = train_height

    data['test']['ref'] = test_ref
    data['test']['width'] = test_width
    data['test']['height'] = test_height

    print(f"length of train : {len(train_ref)}")
    print(f"length of test : {len(test_ref)}")

    with open('./pretrain_cartoon_split.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    json_dump('./pretrain_cartoon_info.json')
