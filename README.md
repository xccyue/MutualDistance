# MutualDistance
[ECCV 2024] Official PyTorch implementation of the paper "Scene-aware Human Motion Forecasting via Mutual Distance Prediction"


### Datasets
#### GTM-IM
For the original dataset please contact the authors of [Long-term Human Motion Prediction with Scene Context](https://zhec.github.io/hmp/).

For the motion sequences used in paper are from [ContAwareMotionPred](https://github.com/wei-mao-2019/ContAwareMotionPred).

After downloading the dataset, please update the path of motion sequence, scene sdf and scene points in the code.

We also provide the preprocessed scene sdf file and checkpoints on [google drive](https://drive.google.com/drive/folders/1QyW6gSJdd2KVerNPAS8xf_wOU06OrNyn?usp=sharing)

After downloading the checkpoints, replace the folder with same name under the GTAIM.

### Training on GTA-IM
Please open the corresponding folder and run 

```
python stage1/train_motion.py 
```

```
python stage2/train_motion.py 
```

```
python finalmodel/train_motion.py --resume_model_s1 xxx/MutualDistance/GTAIM/checkpoints/stage1 --resume_model_s2 xxx/MutualDistance/GTAIM/checkpoints/stage2
```

xxx is the path on your systerm.

### Evaluation on GTA-IM
Please open the corresponding folder and run 

```
python finalmodel/test_motion.py --resume_model xxx/MutualDistance/GTAIM/checkpoints/final
```

xxx is the path on your systerm.



### Datasets
#### HUMANISE

For the original dataset please contact the authors of [HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes](https://github.com/Silverster98/HUMANISE).

Please also download the [SMPL-X] (https://smpl-x.is.tue.mpg.de/download.php), here we use the v1.1 version.

After downloading the dataset, please update the path of pure_motion_folder and align_data_folder in the code.


We also provide the preprocessed scene sdf file, mutual distance file and checkpoints on [google drive](https://drive.google.com/drive/folders/1TBXVSvFVO5kyqBZXnpvfFIwfpDMxM-Kp?usp=sharing)

After downloading, please update the paths in the configuration file.

### Training on HUMANISE
Please open the corresponding folder and run 

```
python stage1/train_motion.py 
```

```
python stage2/train_motion.py 
```

```
python finalmodel/train_motion.py --resume_model_s1 xxx/MutualDistance/GTAIM/checkpoints/stage1 --resume_model_s2 xxx/MutualDistance/GTAIM/checkpoints/stage2
```

xxx is the path on your systerm.

### Evaluation on HUMANISE
Please open the corresponding folder and run 

```
python finalmodel/test_motion.py --resume_model xxx/MutualDistance/HUMANISE/checkpoints/final
```

xxx is the path on your systerm.

### Re-training of ContAware and STAG
Please adpot our optimizer to the second stage (Motion Prediction) of ContAware and STAG.




## 📝 TODO List
- [Y] Data preparation.
- [Y] Release training and evaluation codes.
- [Y] Release checkpoints.


### Acknowledgments

The overall code framework (dataloading, training, testing etc.) is adapted from 
[DLow](https://github.com/Khrylx/DLow)
[ContAwareMotionPred](https://github.com/wei-mao-2019/ContAwareMotionPred) 
[HUMANISE](https://github.com/Silverster98/HUMANISE)

