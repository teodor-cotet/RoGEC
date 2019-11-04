USR_DIR=$PWD/gec_ro/
PROBLEM=gec_ro
DATA_DIR=$PWD/gec_ro/data/
TMP_DIR=$PWD/gec_ro/tmp/
MODEL=transformer
#HPARAMS=transformer_base_single_gpu
HPARAMS=transformer_tiny
TRAIN_DIR=$PWD/gec_ro/train/$MODEL-$HPARAMS
mkdir -p $TRAIN_DIR $TMP_DIR $DATA_DIR
DECODE_FILE=$USR_DIR/decode_this_sample.txt
RESULTS_DECODE_FILE=test_corrected_wiki.txt
touch $RESULTS_DECODE_FILE


BEAM_SIZE=4
ALPHA=0.6


t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$RESULTS_DECODE_FILE

cat $RESULTS_DECODE_FILE