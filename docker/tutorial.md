# mmv_im2im Docker Deployment
## Installation
### Prerequisite:
- You need to download the docker for your operating system, see the tutorial [here](https://docs.docker.com/get-docker/). 
- To utilize GPU, it is also required to install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit). 
- on MacOS, we recommend to allocate at least 8GB memory and 4GB swap for docker to run our example.
### 1. Arm64(Apple M1/2)
Firstly, pull our image from the dockerhub:
```bash
docker pull mmvlab/mmv_im2im:v0.4.0_arm64
```
Then create and run a container:
```bash
# make sure you are in the root dir of mmv_im2im package
bash docker/arm64/run_container.sh v0.4.0_arm64
```
### 2. Amd64(Intel/AMD CPU)
Firstly, pull our image from the dockerhub:
```bash
docker pull mmvlab/mmv_im2im:v0.4.0_amd64
```
Then create and run a container:
```bash
# make sure you are in the root dir of mmv_im2im package
bash docker/amd64/run_container.sh v0.4.0_amd64
```
### 3. CUDA(Nvidia GPU)
Firstly, pull our image from the dockerhub:
```bash
docker pull mmvlab/mmv_im2im:v0.4.0_amd64_cu113
```
Then create and run a container:
```bash
# make sure you are in the root dir of mmv_im2im package
bash docker/cuda/run_container.sh v0.4.0_amd64_cu113
```

## Simple tutorial: labelfree 2d task
We illustrate the usability of our package through a simple labelfree 2d task. 
- To download the example data, please refer to this [notebook](../paper_configs/prepare_data/labelfree_2d.ipynb). Please make sure the data is in the right path.
- We recommend you to run the docker using [vscode](https://code.visualstudio.com/) with [docker plugin](https://code.visualstudio.com/docs/containers/overview).
- To run the code:
    - for training:
    ```bash
    run_im2im --config_path 'paper_configs/labelfree_2d_FCN_train.yaml'\
      --data.data_path 'data/labelfree2D/train'\
      --trainer.params "{'max_epochs':10,'accelerator':'auto'}"\
      --data.dataloader.train.dataloader_params "{'batch_size':1,'num_workers':1}"
    ```
    - for testing:
    ```bash
    run_im2im --config_path 'paper_configs/labelfree_2d_FCN_inference.yaml'\
      --data.inference_input.dir 'data/labelfree2D/test'\
      --data.inference_output.path 'data/labelfree2D/pred'\
      --model.checkpoint 'lightning_logs/version_0/checkpoints/best.ckpt'
    ``` 
