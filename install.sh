# install.sh
# Install the evaluation tw_rouge package
sudo apt-get install git
git clone https://github.com/deankuo/ADL24-HW2.git
cd ADL24-HW2
pip install -e tw_rouge

# Install the required packages
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
