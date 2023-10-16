## install
following [parsenet](https://github.com/Hippogriff/parsenet-codebase#installation) install cmds


## data prepare

- please download our dataset from [Baiduyun](https://pan.baidu.com/s/1apCmf8Xa_rXyRdWl4ybJpg?pwd=meta) (password is *meta*) and load all files of folder *sed_net_data* to *data* 
- please download parsenet datasets from [Baiduyun](https://pan.baidu.com/s/16fggrr-qQRc2yu6ECQNaoA) (password is *meta*) and load all files of folder *parsenet* to *data_parsenet* 
- you can download our pretained models from [Baiduyun](https://pan.baidu.com/s/1rMMD_0VaOGTmpMcIozjp3Q) (password is *meta*) and load all weights of folder *ckpts* to *ckpts* 


## train & test
- test model with normal

```python 
python generate_predictions_aug.py configs/config_SEDNet_normal.yml NoSave no_multi_vote no_fold5drop
```

- train model with normal

```python 
python train_sed_net.py configs/config_SEDNet_normal.yml
```

## reference
1. [parsenet](https://github.com/Hippogriff/parsenet-codebase)
2. [hpnet](https://github.com/SimingYan/HPNet)