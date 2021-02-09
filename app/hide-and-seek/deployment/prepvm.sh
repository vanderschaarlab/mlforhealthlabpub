# Installs all the requirements for a competition worker VM and performs other prep.
# NOTE:
# Script needs to run *twice*, as there is a reboot.
# PREREQUISITE:
# The competition-containing repository must be at $REPO_ROOT.

# SET VARIABLES:
REPO_ROOT="/home/ubuntu/mlforhealthlabcode"
BROKER_URL="<ENTER>"
STAR_CENTRE_SSH_KEY="<ENTER>"



# ----------------------------------------------------------------------------------------------------------------------

# 1. Install Nvidia drivers.
if ! [ -x "$(command -v nvidia-smi)" ]; then
  echo "1. Install Nvidia drivers." ; echo ""
  sudo apt-get update ; sudo apt-get upgrade -y 
  sudo apt install nvidia-driver-450 -y
  sudo reboot
fi
echo "1. Install Nvidia drivers: Testing..." ; echo ""
nvidia-smi # Test
echo "1. Install Nvidia drivers: DONE." ; echo ""
 
# 2. Install Docker.
echo "2. Install Docker." ; echo ""
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y

sudo docker run hello-world # Test

# To run `docker` without sudo:
sudo groupadd docker ; sudo usermod -aG docker $USER ; newgrp docker <<EOF
docker run hello-world # Test
EOF
echo "2. Install Docker: DONE." ; echo ""

# 3. Install Nvidia-docker.
echo "3. Install Nvidia-docker." ; echo ""
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2 -y
sudo systemctl restart docker 

sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi # Test
echo "3. Install Nvidia-docker: DONE." ; echo ""

# 4. Preapre for file sync between VMs.
echo "4. Preapre for file sync between VMs." ; echo ""
sudo apt-get install unison -y  # On all VMs.

sudo su <<EOF
cd ~/.ssh
echo $STAR_CENTRE_SSH_KEY > authorized_keys # Not using >> as su authorized_keys on Azure VM already contains some stuff that we overwrite.
echo "/root/.ssh/authorized_keys :"
cat authorized_keys
EOF
echo "4. Preapre for file sync between VMs: DONE." ; echo ""

# 5. Prepare and launch compute_worker.
echo "5. Prepare and launch compute_worker." ; echo ""
# Prepare .env:
cd $REPO_ROOT/app/hide-and-seek/competitions-v1-compute-worker
cp .env_hns .env
sed -i "s?<YOUR_pyamqp_URL>?$BROKER_URL?" .env
echo "$REPO_ROOT/app/hide-and-seek/competitions-v1-compute-worker/.env :"
cat .env # Test
# Launch compute_worker:
newgrp docker <<EOF
bash run.sh
EOF
echo "5. Prepare and launch compute_worker: DONE." ; echo ""

newgrp docker
