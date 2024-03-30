LLAVA=/home/yiheng/code/LLaVA
TRANSFORMERS=/home/yiheng/code/transformers
COCO=/media/drive_16TB/data/coco
POPE=/home/yiheng/code/POPE
MODEL=/media/drive_16TB/huggingface
PRETRAINED=/media/drive_16TB/model/llava_pretrained

docker run --runtime nvidia -it --shm-size 32g --name llava \
-v $LLAVA:/workspace/LLaVA \
-v $TRANSFORMERS:/workspace/transformers \
-v $MODEL:/workspace/huggingface \
-v $PRETRAINED:/workspace/pretrained \
-v $COCO:/workspace/data/coco \
-v $POPE:/workspace/POPE \
llava:v1.1