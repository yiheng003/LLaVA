
root="/workspace"
model_root="${root}/huggingface"
# model='llava-v1.6-vicuna-7b'
model='llava-v1.5-7b'
pope_subset='adversarial'
lvis_subset='6e4'

python3 model_vqa_in_context_learning.py \
       --model-path ${model_root}/${model} \
       --question-file ${root}/LLaVA/llava/eval/in_context/adversarial_question_small.json \
       --image-folder ${root}/data/coco \
       --answers-file ${root}/LLaVA/llava/eval/in_context/test_img_neg_img.json \
       --demonstration_file ${root}/POPE/output/in_context/lvis_3_shot_demo.json \
       --temperature 0 \
       --conv-mode vicuna_v1

# python3 lvis_eval_pope_coco2lvis.py --annotation-file ${root}/LLaVA/llava/eval/in_context/adversarial_question_small.json \
# 	--question-file ${root}/LLaVA/llava/eval/in_context/adversarial_question_small.json \
# 	--result-file ${root}/LLaVA/llava/eval/in_context/test_img_neg_img.json
