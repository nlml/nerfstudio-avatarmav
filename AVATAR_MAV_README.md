### Nov 15

Writing commands below

```bash
# TODO
# [x] Proper validation set loading and eval
# [x] Ability to render a set of images given a transforms.json
# [ ] Add FG mask loss
# [x] Dont load all ims into memory!
# [x] Gives eyeballs pose to expression param
# [ ] Deformation field to proposal network?

#### Config for no proposal sampler:
ns-train avatarmav \
    --output-dir disablePropNets_aabbCollider_128nerfSamples_fixSceneScale \
    --pipeline.model.use-aabb-box-collider True \
    --pipeline.model.disable-proposal-nets True \
    --pipeline.model.num-nerf-samples-per-ray 128 \
    --viewer.websocket-port 6009 \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.eval-num-images-to-sample-from 32 \
    --experiment-name $EXP_NAME \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 64 \
    --pipeline.model.headmodule-deform-bs-res 64 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data $JSON_PATH \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.3 \
    --disable-scaling True


#### This is the general command used for final training runs:
#!/usr/bin/env bash
echo "Start 074..."
ns-train avatarmav \
    --output-dir fixed-intrinsics \
    --viewer.websocket-port 6009 \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.eval-num-images-to-sample-from 32 \
    --experiment-name 074-UNION \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 64 \
    --pipeline.model.headmodule-deform-bs-res 64 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/074/UNION_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2 \
|| true
echo "Done 074"


#### This is the eval script for train and validation sets:
#!/usr/bin/env bash
# eval_many.sh

USER_ID=$1

# Ensure arg was provided:
if [ -z "$USER_ID" ]; then
    echo "Usage: ./eval_many.sh <USER_ID>"
    exit 1
fi

RUN_ID=$USER_ID"-UNION"

JSON_PATH="nersemble_masked/"$USER_ID"/UNION_"$USER_ID"_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json"
./eval_script.sh $RUN_ID $JSON_PATH || true

JSON_PATH="nersemble_masked/"$USER_ID"/UNION_"$USER_ID"_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_test.json"


#### Where below is eval_script.sh:
#!/usr/bin/env bash
# eval_script.sh

# ./eval_script.sh 253-UNION nersemble_masked/253/253_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json

RUN_NAME=$1
EVAL_TRANSFORMS_JSON_PATH=$2

# Ensure both arguments were provided:
if [ -z "$RUN_NAME" ] || [ -z "$EVAL_TRANSFORMS_JSON_PATH" ]; then
    echo "Usage: ./eval_script.sh <RUN_NAME> <EVAL_TRANSFORMS_JSON_PATH>"
    exit 1
fi

RUN_DIR=$(ls -td fixed-intrinsics/$RUN_NAME/avatarmav/*/ | head -1)
CFG_PATH=$RUN_DIR"config.yml"
MDLS_PATH=$RUN_DIR"nerfstudio_models"

echo $CFG_PATH
echo $MDLS_PATH

OUTDIR="fixed-intrinsics/eval_outputs/$RUN_NAME/$EVAL_TRANSFORMS_JSON_PATH"
mkdir -p $OUTDIR

EVAL_TRANSFORMS_JSON_PATH=$EVAL_TRANSFORMS_JSON_PATH ns-eval --load-config $CFG_PATH --output-path $OUTDIR/results.json --render-output-path $OUTDIR/images

#### After running the above, `cp get_results_table.py /path/to/data/dir/fixed-intrinsics/eval_outputs/`
#### then run `python get_results_table.py` to get the paper results tables and images.

#### Commands below are from general experimenting:

# Try without proposal nets
ns-train avatarmav \
    --output-dir outputs-no-proposal-nets \
    --viewer.websocket-port 6009 \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.eval-num-images-to-sample-from 32 \
    --experiment-name 302-UNION \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.num-proposal-iterations 0 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 64 \
    --pipeline.model.headmodule-deform-bs-res 64 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/302/UNION_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2


# MAIN TRAINING RUN FOR 074 (plus 302,304,253,264,218)
ns-train avatarmav \
    --viewer.websocket-port 6009 \
    --viewer.quit-on-train-completion True \
    --pipeline.datamanager.eval-num-images-to-sample-from 32 \
    --experiment-name 074-UNION \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 64 \
    --pipeline.model.headmodule-deform-bs-res 64 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/074/UNION_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2

# TESTING fg_mask loss
# Reduced the train_num_rays_per_batch to 4096, exp-dim to 48, bs-res to 32
ns-train avatarmav \
    --vis tensorboard \
    --viewer.websocket-port 7860 \
    --experiment-name UNION_neck2pose2cam_L1-lessGPUMem-fgmasktest \
    --pipeline.datamanager.eval-num-images-to-sample-from 16 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 48 \
    --pipeline.model.headmodule-deform-bs-res 32 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/UNION_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2

# For rendering images
ns-eval --load-config outputs/UNION_neck2pose2cam_L1-continue/avatarmav/2023-11-15_215209/config.yml --output-path test_ns_eval.json --render-output-path test_ns_eval_dir


# Union sequence - continue with VanillaDataMgr
ns-train avatarmav \
    --vis tensorboard \
    --load-dir outputs/UNION_neck2pose2cam_L1/avatarmav/2023-11-15_175043/nerfstudio_models \
    --pipeline.datamanager.eval-num-images-to-sample-from 32 \
    --experiment-name UNION_neck2pose2cam_L1 \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 64 \
    --pipeline.model.headmodule-deform-bs-res 64 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/UNION_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2


### Test for Davide
# Reduced the train_num_rays_per_batch to 4096, exp-dim to 48, bs-res to 32
ns-train avatarmav \
    --vis viewer+tensorboard \
    --viewer.websocket-port 7860 \
    --experiment-name UNION_neck2pose2cam_L1-lessGPUMem \
    --pipeline.datamanager.eval-num-images-to-sample-from 16 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --pipeline.datamanager.pixel-sampler.num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 48 \
    --pipeline.model.headmodule-deform-bs-res 32 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/UNION_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2


# Union sequence
ns-train avatarmav \
    --vis tensorboard \
    --pipeline.datamanager.eval-num-images-to-sample-from 16 \
    --experiment-name UNION_neck2pose2cam_L1 \
    --pipeline.datamanager.train-num-cameras-per-batch 32 \
    --pipeline.model.num-cameras-per-batch 32 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.headmodule-exp-dim 64 \
    --pipeline.model.headmodule-deform-bs-res 64 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/UNION_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --neck-pose-to-expr True \
    --eyes-pose-to-expr True \
    --scene-scale 0.2



# # Test for Union sequence with medium sequence
# ns-train avatarmav \
#     --vis tensorboard \
#     --pipeline.datamanager.eval-num-images-to-sample-from 16 \
#     --experiment-name testdel \
#     --pipeline.datamanager.train-num-cameras-per-batch 32 \
#     --pipeline.model.num-cameras-per-batch 32 \
#     --pipeline.model.use-l1-loss True \
#     --pipeline.model.headmodule-feature-res 128 \
#     --pipeline.model.headmodule-exp-dim 64 \
#     --pipeline.model.headmodule-deform-bs-res 64 \
#     --optimizers.fields.optimizer.lr 5e-3 \
#     --optimizers.proposal_networks.optimizer.lr 1e-3 \
#     --pipeline.model.background-color white \
#     nerfstudio-data \
#     --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
#     --apply-neck-rot-to-flame-pose True \
#     --apply-flame-poses-to-cams True \
#     --neck-pose-to-expr True \
#     --eyes-pose-to-expr True \
#     --scene-scale 0.2



ns-train avatarmav \
    --vis tensorboard \
    --pipeline.datamanager.eval-num-images-to-sample-from 64 \
    --experiment-name neck2pose2cam_L1_tb \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --scene-scale 0.2


# Short run
ns-train avatarmav \
    --vis tensorboard \
    --experiment-name neck2pose2cam_L1_30kits_tb \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 64 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --scene-scale 0.25


## THIS WAS THE BEST SO FAR!
ns-train avatarmav \
    --viewer.websocket-port 6789 \
    --load-dir outputs/neck2pose2cam_L1/avatarmav/2023-11-15_132219/nerfstudio_models \
    --experiment-name neck2pose2cam_L1 \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.use-l1-loss True \
    --pipeline.model.headmodule-feature-res 128 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --scene-scale 0.2


ns-train avatarmav \
    --experiment-name defaultish_neck2pose \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.headmodule-feature-res 128 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-neck-rot-to-flame-pose True \
    --apply-flame-poses-to-cams True \
    --scene-scale 0.2


# Testing with an already-trained model
ns-train avatarmav \
    --pipeline.datamanager.train-num-rays-per-batch 65536 \
    --optimizers.fields.optimizer.lr 1e-1 \
    --max-num-iterations 600000 \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --viewer.websocket-port 6009 \
    --load-dir long-run/unnamed/avatarmav/2023-11-13_220123/nerfstudio_models/nerfstudio_models \
    --pipeline.model.background-color white nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --apply-flame-poses-to-cams True \
    --scene-scale 0.2


# To set:
#     num_proposal_samples_per_ray: Tuple[int, ...] = (64,)
# num_proposal_iterations: int = 1
# use_same_proposal_network: bool = False
#     use_avatarmav_field_as_proposal_network: bool = True

ns-train avatarmav \
    --viewer.websocket-port 6009 \
    --experiment-name defaultish \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.headmodule-feature-res 128 \
    --optimizers.fields.optimizer.lr 1e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-2 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --scene-scale 0.2

# Re-enable dist and interlevel losses
ns-train avatarmav \
    --viewer.websocket-port 7860 \
    --experiment-name regLoss1e-3_etc_lowerLR \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.headmodule-feature-res 128 \
    --optimizers.fields.optimizer.lr 1e-3 \
    --optimizers.proposal_networks.optimizer.lr 1e-4 \
    --pipeline.model.use-avatarmav-field-as-proposal-network True \
    --pipeline.model.use-same-proposal-network True \
    --pipeline.model.num-proposal-iterations 1 \
    --pipeline.model.num-proposal-samples-per-ray 64 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.proposal-update-every 10000 \
    --pipeline.model.proposal-warmup 1 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --scene-scale 0.2

ns-train avatarmav \
    --viewer.websocket-port 6009 \
    --experiment-name regLoss1e-3_etc \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.interlevel-loss-mult 1.0 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.headmodule-feature-res 128 \
    --optimizers.fields.optimizer.lr 1e-2 \
    --optimizers.proposal_networks.optimizer.lr 1e-3 \
    --pipeline.model.use-avatarmav-field-as-proposal-network True \
    --pipeline.model.use-same-proposal-network True \
    --pipeline.model.num-proposal-iterations 1 \
    --pipeline.model.num-proposal-samples-per-ray 64 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.proposal-update-every 10000 \
    --pipeline.model.proposal-warmup 1 \
    --pipeline.model.background-color white \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --scene-scale 0.2


ns-train avatarmav \
    --viewer.websocket-port 6009 \
    --experiment-name regLoss1e-3_hmFtRes128_4camPerBtch_lr1e-3 \
    --pipeline.model.background-color white \
    --pipeline.model.distortion-loss-mult 0.0 \
    --pipeline.model.interlevel-loss-mult 0.0 \
    --pipeline.model.num-proposal-samples-per-ray 64 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.proposal-update-every 10000 \
    --pipeline.model.proposal-warmup 1 \
    --optimizers.fields.optimizer.lr 1e-3 \
    --pipeline.datamanager.train-num-cameras-per-batch 4 \
    --pipeline.model.num-cameras-per-batch 4 \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --scene-scale 0.2



ns-train avatarmav \
    --viewer.websocket-port 6009 \
    --experiment-name regLoss1e-3_hmFtRes128_16camPerBtch_lr5e-3 \
    --pipeline.model.background-color white \
    --pipeline.model.distortion-loss-mult 0.0 \
    --pipeline.model.interlevel-loss-mult 0.0 \
    --pipeline.model.num-proposal-samples-per-ray 64 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.offset-reg-loss 1e-3 \
    --pipeline.model.headmodule-feature-res 128 \
    --pipeline.model.proposal-update-every 10000 \
    --pipeline.model.proposal-warmup 1 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --pipeline.datamanager.train-num-cameras-per-batch 16 \
    --pipeline.model.num-cameras-per-batch 16 \
    nerfstudio-data \
    --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json \
    --scene-scale 0.2

```

### Nov 14

Training on a longer sequence works here, along with expression control, but the results are quite blurry.

```
ns-train avatarmav --pipeline.model.background-color white nerfstudio-data --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json --scene-scale 0.2
```

### Nov 12, first version working at this commit with following command:

```
ns-train avatarmav --pipeline.model.background-color white nerfstudio-data --data nersemble_masked/104/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train_1_timestep.json --scene-scale 0.2
```
