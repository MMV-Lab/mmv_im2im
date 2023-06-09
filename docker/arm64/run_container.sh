#!/bin/bash
docker run -it \
--rm \
--name mmv_im2im \
--shm-size=8gb \
-v $(pwd)/data:/workspace/mmv_im2im/data \
mmv_im2im \
/bin/bash