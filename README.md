# NRCVQA

Xiaoxi Yang; Xiaojing Chen; Yuan Chen; Yang Zhao; No-Reference Quality Assessment for Cartoon-Like Videos, IEEE International Conference on Multimedia and Expo, 2025.( Best Student Paper Nomination Award)

# Requirement

# Create Environment

```
conda create -n NRCVQA python=3.8
conda activate NRCVQA
```

# Install Pytorch and Tensorboard

```
# pytorch: 1.8.2
# CUDA 11.1 

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

```
conda install tensorboard
```

# Install the Requirements

```
pip install -r requirements.txt
```


# Self-supervised Cartoon10K and Fine-tuned Database

We selected 300 original cartoon videos from Internet. We also provided the 300 original vidoes and the code for generation distortion videos.

# 300 original videos and fine-tune cartoon videos

[DownLoad Link](The download link for the 600 original and fine-tuned cartoon videos can be obtained by contacting thr corresponding author.)


# Distortion generate

`./pretrained_database/make_dis.py`

# Pre-train

`python pre_trained.py --batch-size=4 --batch-test=4 --frame=12 --model=pre_train --epoch=60 --base_lr=5e-5 `
 
# Fine-tuning

- change the **video path** in `getVQA.py`. e. g. `'/mnt/wwn-0x5000cca0c3e1998a/fine_tune_database/Syn_videos'`

- train-test in Syn_cartoon Database: 
`python finetuned_syn.py --batch-size=6 --batch-test=6 --dataset=Syn_CVQA --frame=12 --base_lr=5e-5 --loss=plcc --fine_tune=True --best=0.8 --idx=0`

- train-test in Real_cartoon Database: 
`python finetuned_true.py --batch-size=6 --batch-test=6 --dataset=True_CVQA --frame=12 --base_lr=5e-5 --loss=plcc --fine_tune=True --best=0.8 --idx=0`


## Demo

run `python demo.py` to get the predict for one test video. (You can modify the setting by yourself.)


# Finetuning Datasets:
https://pan.baidu.com/s/1mOo-ytPlmeCO6S81y5jVuw  code: 7gg7 




