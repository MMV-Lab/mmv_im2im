#!/bin/bash
docker run -it \
--rm \
--name mmv_im2im_amd64 \
--shm-size=2gb \
-v $(pwd)/data:/workspace/mmv_im2im/data \
mmv_im2im:amd64 \
/bin/bash