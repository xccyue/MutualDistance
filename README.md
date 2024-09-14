# MutualDistance
[ECCV 2024] Official PyTorch implementation of the paper "Scene-aware Human Motion Forecasting via Mutual Distance Prediction"


### Datasets
#### GTM-IM
For the original dataset please contact the authors of [Long-term Human Motion Prediction with Scene Context](https://zhec.github.io/hmp/).

For the motion sequences used in paper are from [ContAwareMotionPred](https://github.com/wei-mao-2019/ContAwareMotionPred).

After downloading the dataset, please update the path of motion sequence, scene sdf and scene points in the code.

We also provide the preprocessed sdf file on [google drive](https://drive.google.com/drive/folders/1QyW6gSJdd2KVerNPAS8xf_wOU06OrNyn?usp=sharing)

After downloading the checkpoints, replace the folder with same name under the GTAIM.


### Evaluation on GTA-IM
Please open the corresponding folder and run 

```
python finalmodel/test_motion.py --resume_model xxx/MutualDistance/GTAIM/checkpoints/final
```

xxx is the path on your systerm.

The code will be released soon.
## 📝 TODO List
- [ ] Data preparation.
- [ ] Release training and evaluation codes.
- [ ] Release checkpoints.


### Acknowledgments

The overall code framework (dataloading, training, testing etc.) is adapted from 
[DLow](https://github.com/Khrylx/DLow)
[ContAwareMotionPred](https://github.com/wei-mao-2019/ContAwareMotionPred) 
[HUMANISE](https://github.com/Silverster98/HUMANISE)

