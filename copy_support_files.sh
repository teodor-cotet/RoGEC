echo "copying files to $1"
scp checkpoints/10m_transformer_768/tokenizer_ro.subwords teodor.cotet@$1:~/RoGEC/checkpoints/10m_transformer_768/
scp checkpoints/10m_transformer_768/tokenizer_ro.subwords teodor.cotet@$1:~/RoGEC/checkpoints/10m_transformer_768_finetune/
scp checkpoints/10m_transformer_768/tokenizer_ro.subwords teodor.cotet@$1:~/RoGEC/checkpoints/10m_transformer_768_retrain/
scp  -r corpora/cna/ teodor.cotet@$1:~/RoGEC/corpora/cna/
scp "/media/teo/drive hdd/gec/corpora/wiki_synthetic/arpa/30m_wiki_clean.zip" teodor.cotet@$1:~/RoGEC/corpora/