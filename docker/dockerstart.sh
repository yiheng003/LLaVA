LLAVA=/home/yiheng/code/LLaVA_cip
LVIS=/media/drive_16TB/data/coco
POPE=/media/drive_16TB/POPE/
MODEL=/media/drive_16TB/huggingface
LLAVA_CKPT=/media/drive_16TB/model/llava_pretrained

docker run --runtime nvidia -it --shm-size 32g --name llava \
-v $LLAVA:/workspace/LLaVA \
-v $MODEL:/workspace/huggingface \
-v $LLAVA_CKPT:/workspace/model \
-v $LVIS:/workspace/data/coco \
-v $POPE:/workspace/POPE \
llava:pip_transformer
