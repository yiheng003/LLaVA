root="/home/yiheng/code"
pope_subset='adversarial'
python3 eval_pope_semantic_struct.py --annotation-file ${root}/POPE/output/coco/coco_pope_${pope_subset}.json \
	--question-file ${root}/POPE/output/coco/coco_pope_${pope_subset}.json \
	--result-file ${root}/POPE/output/semantic_structure/coco_pope_${pope_subset}_ans.json \
    --graph ${root}/POPE/output/semantic_structure/TP_${pope_subset}.png