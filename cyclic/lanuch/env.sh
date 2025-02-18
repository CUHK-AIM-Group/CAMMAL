# pip uninstall taming-transformers
# pip install pytorch-lightning==1.0.8

. ~/tool/anaconda3/etc/profile.d/conda.sh
conda activate linkbert2
# cp ~tmp/download/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints
cd ~/code/img2text2img/llm-cxr

################## nlpc_sys ##################
yum install java-1.6.0-openjdk

sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version

source ~/tool/environment/linkbert2/bin/activate
cd ~/code/image2text2image/CXR2Report2CXR

mkdir -p /root/.cache/torch/hub/checkpoints
cp ~/download/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints



pip install datasets==2.10.0
pip install deepspeed==0.8.3


Use V100 gpu:

sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version

. /apdcephfs/share_916081/jarviswang/tool/anaconda3/etc/profile.d/conda.sh
conda activate linkbert2V100
cd ~code/img2text2img/llm-cxr


eutil>=2.8.1->pandas->datasets) (1.16.0)
Installing collected packages: huggingface-hub
  Attempting uninstall: huggingface-hub
    Found existing installation: huggingface-hub 0.14.1
    Uninstalling huggingface-hub-0.14.1:
      Successfully uninstalled huggingface-hub-0.14.1
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following
 dependency conflicts.
transformers 4.28.1 requires huggingface-hub<1.0,>=0.11.0, but you have huggingface-hub 0.0.19 which is incompatible.
Successfully installed huggingface-hub-0.0.19'


-----------------environment for encoding image features--------------------

use V100 gpu

vi /root/.bashrc

# input
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('~software/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "~software/mambaforge/etc/profile.d/conda.sh" ]; then
        . "~software/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="~software/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "~software/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "~software/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<

source /root/.bashrc
mamba activate taming
cd ~code/PLM/taming-transformers
cp ~tmp/download/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints