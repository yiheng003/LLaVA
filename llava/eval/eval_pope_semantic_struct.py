import os
import json
import argparse
import matplotlib.pyplot as plt

def eval_pope(answers, label_file, out_graph):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    similarity_list = [float(answer['patch_wise_similarity']) for answer in answers]

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
    similarity_tp = []
    similarity_fp = []
    similarity_tn = []
    similarity_fn = []

    for pred, label, similarity in zip(pred_list, label_list, similarity_list):
        if pred == pos and label == pos:
            TP += 1
            similarity_tp.append(similarity)
        elif pred == pos and label == neg:
            FP += 1
            similarity_fp.append(similarity)
        elif pred == neg and label == neg:
            TN += 1
            similarity_tn.append(similarity)
        elif pred == neg and label == pos:
            FN += 1
            similarity_fn.append(similarity)

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

    tp_value = [4 for _ in range(len(similarity_tp))]
    tn_value = [3 for _ in range(len(similarity_tn))]
    fp_value = [2 for _ in range(len(similarity_fp))]
    fn_value = [1 for _ in range(len(similarity_fn))]

    plt.scatter(similarity_tp, tp_value)
    plt.scatter(similarity_tn, tn_value)
    plt.scatter(similarity_fp, fp_value)
    plt.scatter(similarity_fn, fn_value)
    plt.title('similarity')
    plt.savefig(out_graph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--graph", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]

    category = args.annotation_file[:-5]
    cur_answers = [x for x in answers]
    print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
    eval_pope(cur_answers, args.annotation_file, args.graph)
    print("====================================")
