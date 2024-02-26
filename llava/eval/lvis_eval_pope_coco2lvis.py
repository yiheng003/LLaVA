import os
import json
import argparse

def eval_pope(answers, label_file, subset_coco=False):
    lvis_data = json.load(open('lvis_v1_val.json'))
    coco_data = json.load(open('coco_to_synset.json'))
    lvis_categories = lvis_data['categories']
    coco_synsets = [coco_data[d]['synset'] for d in coco_data]
    for category in lvis_categories:
        category['name'] = category['name'].replace('_', ' ')
        if category['synset'] in coco_synsets:
            category['is_coco'] = True
        else:
            category['is_coco'] = False
    
    # index {lvis_categories} with 'name'
    lvis_categories = {category['name']: category for category in lvis_categories}
    question_list = [json.loads(q) for q in open(label_file, 'r')]
    question_list = {question['question_id']: question['text'].replace('Is there a ', '').replace('Is there an ', '').replace(' in the image?', '').replace(' in the imange?', '') for question in question_list}
    question_ids = []
    for question_id, question_category in question_list.items():
        assert question_category in lvis_categories
        if subset_coco:
            subset = 'coco'
            if lvis_categories[question_category]['is_coco']:
                question_ids.append(question_id)
        else:
            subset = 'lvis'
            if not lvis_categories[question_category]['is_coco']:
                question_ids.append(question_id)

    print(f'# {subset} questions: {len(question_ids)}')
    # todo valid question ids
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r') if json.loads(q)['question_id'] in question_ids]
    answers = [answer for answer in answers if answer['question_id'] in question_ids]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    
    category = args.annotation_file[10:-5]
    print('# samples: {}'.format(len(answers)))
    eval_pope(answers, args.annotation_file, subset_coco=True)
    print("====================================")
    eval_pope(answers, args.annotation_file, subset_coco=False)
    print("====================================")
