git clone https://github.com/TeodorMihai/RoGEC.git
pip3 install bert-for-tf2
pip3 install google-cloud-storage
cd RoGEC 
mkdir -p checkpoints/10m_transformer_768_finetune
mkdir -p checkpoints/10m_transformer_768_retrain
mkdir -p checkpoints/10m_transformer_768
mkdir -p corpora/cna/
cd ../
git clone https://github.com/kpu/kenlm.git
cd kenlm
sudo apt-get -y install cmake
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
mkdir -p build
cd build
cmake ..
make -j 4
pip3 install https://github.com/kpu/kenlm/archive/master.zip