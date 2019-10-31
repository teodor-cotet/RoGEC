USR_DIR=$PWD/gec_ro/
PROBLEM=gec_ro
DATA_DIR=$PWD/gec_ro/data/
TMP_DIR=$PWD/gec_ro/tmp/
MODEL=transformer
#HPARAMS=transformer_base_single_gpu
HPARAMS=transformer_tiny
TRAIN_DIR=$PWD/gec_ro/train/$MODEL-$HPARAMS
mkdir -p $TRAIN_DIR $TMP_DIR $DATA_DIR


t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR