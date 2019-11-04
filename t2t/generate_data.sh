# put data file in data folder then generate
USR_DIR=$PWD/gec_ro/
PROBLEM=gec_ro
DATA_DIR=$PWD/gec_ro/data/
TMP_DIR=$PWD/gec_ro/tmp/
#HPARAMS=transformer_base_single_gpu
mkdir -p  $DATA_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM