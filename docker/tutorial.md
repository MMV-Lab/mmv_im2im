# mmv_im2im Docker Deployment
## Installation
### 1. Arm64(Apple M1/2)
Firstly, pull our image from the dockerfile:
```bash
docker pull mmvlab/mmv_im2im:v0.4.0_arm64
```
Then create and run a container:
```bash
# make sure you are in the root dir of mmv_im2im package
bash docker/arm64/run_container.sh
```
### 2. Amd64(Intel/AMD CPU)
Firstly, pull our image from the dockerfile:
```bash
docker pull mmvlab/mmv_im2im:v0.4.0_amd64
```
Then create and run a container:
```bash
# make sure you are in the root dir of mmv_im2im package
bash docker/amd64/run_container.sh
```
### 3. CUDA(Nvidia GPU)
Firstly, pull our image from the dockerfile:
```bash
docker pull mmvlab/mmv_im2im:v0.4.0_amd64_cuda113
```
Then create and run a container:
```bash
# make sure you are in the root dir of mmv_im2im package
bash docker/cuda/run_container.sh
```
