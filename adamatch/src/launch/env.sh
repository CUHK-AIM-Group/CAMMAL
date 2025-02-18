

----

source ~/tool/environment/linkbert3/bin/activate

cd ~/code/image2text2image/FLIP_medical

pip install pytorch-lamb

sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version


cd ./ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py

wandb offline

mkdir /root/.cache/torch/
mkdir /root/.cache/torch/hub/
mkdir /root/.cache/torch/hub/checkpoints/
cp ~/download/deit_base_patch16_224-b5f2ef4d.pth /root/.cache/torch/hub/checkpoints/


----
cp ~/tmp2/deit_base_patch16_224-b5f2ef4d.pth /root/.cache/torch/hub/checkpoints/

pip install pytorch-lightning==1.5.10
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
pip install nltk==3.5
pip install pydicom==2.1.2
