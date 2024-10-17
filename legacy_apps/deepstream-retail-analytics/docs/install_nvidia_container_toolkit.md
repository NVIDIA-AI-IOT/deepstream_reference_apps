
# NVIDIA Container Toolkit

* To install NVIDIA container toolkit follow these instructions:

    1. Setup docker repository
    ```bash
    sudo apt-get update

    sudo apt-get install ca-certificates curl gnupg lsb-release

    sudo mkdir -p /etc/apt/keyrings

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

    2. Install [Docker](https://docs.docker.com/engine/install/)
    ```bash
    sudo apt-get update

    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
    ```

    3. Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

    4. **OPTIONAL:** Post-install instructions to run docker without sudo

    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```

    ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    ```
	5. Restart docker 

    ```bash
    sudo systemctl restart docker
    ```