#!/bin/bash
docker run -it \
--rm \
--name mmv_im2im \
--shm-size=2gb \
-v $(pwd)/data:/workspace/mmv_im2im/data \
mmv_im2im:$1 \
/bin/bash