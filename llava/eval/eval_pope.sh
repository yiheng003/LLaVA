root="/workspace"
model_root="${root}/huggingface"
pretrained_root="${root}/pretrained"
# model='llava-v1.6-vicuna-7b'
model='llava-v1.5-7b'
# model='llava-v1.5-7b-lora'
# model='llava-v1.5-7b-voc-lora'
model_base='llava-v1.5-7b'
pope_subset='adversarial'

# python3 model_vqa_cip.py \
#        --model-path ${model_root}/${model} \
#        --question-file ${root}/POPE/output/coco_voc_unseen/coco_voc_seen_pope_${pope_subset}.json \
#        --image-folder ${root}/data/coco/val2014 \
#        --answers-file ${root}/POPE/output/semantic_structure/${model}_coco_voc_seen_pope_${pope_subset}_ans.json \
#        --temperature 0 \
#        --conv-mode vicuna_v1

python3 model_vqa_cip.py \
       --model-path ${model_root}/${model} \
       --question-file ${root}/POPE/output/coco/coco_pope_${pope_subset}.json \
       --image-folder ${root}/data/coco/val2014 \
       --answers-file ${root}/POPE/output/coco/${model}_coco_${pope_subset}_ans.json \
       --temperature 0 \
       --conv-mode vicuna_v1

# python3 model_vqa_cip.py \
#        --model-path ${model_root}/${model} \
#        --model-base ${model_root}/${model_base} \
#        --question-file ${root}/POPE/output/coco_voc_unseen/coco_voc_unseen_pope_${pope_subset}.json \
#        --image-folder ${root}/data/coco/val2014 \
#        --answers-file ${root}/POPE/output/semantic_structure/${model}_coco_voc_unseen_pope_${pope_subset}_ans.json \
#        --temperature 0 \
#        --conv-mode vicuna_v1

# python3 eval_pope_semantic_struct.py --annotation-file ${root}/POPE/output/coco/coco_pope_${pope_subset}.json \
# 	--question-file ${root}/POPE/output/coco/coco_pope_${pope_subset}.json \
# 	--result-file ${root}/POPE/output/coco/${model}_coco_${pope_subset}_ans.json \
#        --graph ${root}/POPE/output/semantic_structure/coco_${model}_${pope_subset}.png
