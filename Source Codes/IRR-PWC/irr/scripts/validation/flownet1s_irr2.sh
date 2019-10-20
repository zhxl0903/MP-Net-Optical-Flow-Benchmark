#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME="saved_check_point/flownet"

# datasets
SINTEL_HOME=(YOUR PATH)/MPI-Sintel-complete/

# model and checkpoint
MODEL=FlowNet1S_irr
CHECKPOINT="$EXPERIMENTS_HOME/FlowNet1S-irr2/checkpoint_best.ckpt"
EVAL_LOSS=MultiScaleEPE_FlowNet_IRR

SIZE_OF_BATCH=4

# validate clean configuration
SAVE_PATH="$EXPERIMENTS_HOME/eval_temp/$MODEL"
python ../../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--num_iters=2 \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS