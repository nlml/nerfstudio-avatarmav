### Nov 14

Training on a longer sequence works here, along with expression control, but the results are quite blurry.

```
ns-train avatarmav --pipeline.model.background-color white nerfstudio-data --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json --scene-scale 0.2
```

### Nov 12, first version working at this commit with following command:

```
ns-train avatarmav --pipeline.model.background-color white nerfstudio-data --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train_1_timestep.json --scene-scale 0.2
```
