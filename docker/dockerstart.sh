LLAVA=/home/yiheng/LLaVA
LVIS=/media/drive_16TB/data/coco
POPE=/home/yiheng/POPE
MODEL=/media/drive_16TB/huggingface

docker run --runtime nvidia -it --shm-size 32g --name llava \
-v $LLAVA:/workspace/LLaVA \
-v $MODEL:/workspace/huggingface \
-v $LVIS:/workspace/data/coco \
-v $POPE:/workspace/POPE \
llava:v1.1
