# Low Resource Grammatical Error Correction Using Wikipedia Edits

This repository contains the code and data for: romanian grammatical error correction testes on RONAC corpus.

## Download Data & pre-trained models
Download the language model: [30mil_wiki_lm](https://nextcloud.readerbench.com/index.php/s/A6WpryeETrj7bJ6)
Download the RONACC corpus: [RONACC]https://nextcloud.readerbench.com/index.php/s/9pwymesT5sycxoM]\\
Download the synthetic corpus [10m_synthetic][]\\
Download trained Transformer-based fine-tune model: [transformer-base-fine-tune][https://nextcloud.readerbench.com/index.php/s/CPAS95MNyZGsKas]\\

## Run Experiment

Install python dependencies:\\
`pip3 install -r requirements.txt`\\
If you want to use LM predictions install kenlm libraries: [kenlm](https://github.com/kpu/kenlm)\\
To run decoding on an existing model run:
`python3 transformer_bert.py --checkpoint=path_to_model_checkpoint --lm_path=path_to_lm --d_model=size_of_model --decode_mode=True`
    (the size of the fine tuned model is 768)
To train models run:
`python3 transformer_bert.py --checkpoint=path_to_model_checkpoint --separate=False --d_model=size_of_model --use_txt=True --dataset_file=path_to_txt_file_wrong_gold --train_mode=True`

If you want to run on tpu, you can use the `--use_tpu=True` argument, but you need to generated tf records file.

## Generating GEC Data with Wiki Edits and ERRANT

The following sections describe how German Wikipedia edits were 
extracted from the Wikipedia revision history and filtered using ERRANT.

If necessary, initialize the submodules containing ERRANT and WikiEdits:

```
git submodule update --init --recursive
```

### ERRANT

#### Install ERRANT
You can use errant normall, just pass the argument -lang ro if you want to use it for Romanian.