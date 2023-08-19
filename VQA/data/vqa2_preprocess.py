import json
from tqdm import tqdm
from collections import Counter
from typing import Dict, List, Tuple


def get_candidate_items(items: List, threshold: int = 0, counter = None) -> Tuple[List, Counter]:
    counter = Counter() if counter is None else counter
    for item in items:
        counter[item] += 1
    res = [item for item, count in counter.items() if count >= threshold]
    return res, counter


def get_candidate_answers4vqa2(data_files, threshold, counter = None, all_answer = False) -> Tuple[List, Counter]:
    # The number of answers appears
    answer_counter = Counter() if counter is None else counter
    answers_list = []
    
    # Type convert
    if not isinstance(data_files, list):
        if isinstance(data_files, str):
            data_files = [data_files, ]
        else:
            raise ValueError('data_files must be str or list')
    
    for filepath in data_files:
        # load VQA-v2 dataset file
        with open(filepath, 'r') as f:
            data = json.load(f)

        for annotation_item in data['annotations']:
            # filter the problem of YES/NO type
            if annotation_item['answer_type'] == 'yes/no': continue
            if all_answer:
                for answer_item in annotation_item['answers']:
                    if answer_item['answer_confidence']  == 'yes':
                        answer = answer_item['answer']
                        answers_list.append(answer)
            else:
                answer = annotation_item['multiple_choice_answer']
                answers_list.append(answer)

    # Get the answer that appears more than the threshold
    candidate_answers, counter = get_candidate_items(answers_list, threshold, counter)
    return candidate_answers, counter


def load_candidate_answer_dict(answer2idx_filepath: str, idx2answer_filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(answer2idx_filepath, 'r') as a2i, open(idx2answer_filepath, 'r') as i2a:
        answer2idx = json.load(a2i)
        idx2answer = json.load(i2a)
    assert len(answer2idx) == len(idx2answer), 'The length of answer2idx must be equal to idx2answer'
    return answer2idx, idx2answer


def make_dataset_file(question_filepath: str, answer_filepath: str, target_filepath: str):
    # load VQA-v2 dataset file
    with open(question_filepath, 'r') as qf, open(answer_filepath, 'r') as af:
        questions_json = json.load(qf)
        annotations_json = json.load(af)
    
    # Map the question id to the question
    questions = questions_json['questions']
    questionid2question = {question['question_id']: question for question in questions}
    
    # Gather the question, answer, and image id
    question_answer_imageid = [] 
    annotations = annotations_json['annotations']
    for annotation in tqdm(annotations, desc = 'Prase annotations'):
        question_id = annotation['question_id']
        answer = annotation['multiple_choice_answer']
        question = questionid2question[question_id]['question']
        image_id = annotation['image_id']
        image_filename = f'{image_id:012d}.jpg'
        question_answer_imageid.append({'id': question_id, 'question': question, 'answer': answer, 'image': image_filename})
        
    # Write to file
    with open(target_filepath, 'w') as f:
        for pair in tqdm(question_answer_imageid, desc = 'Write to file'):
            json.dump(pair, f)
            f.write('\n')


def vqa2_annotation2eval_annotation(vqa2_annotation_filepath: str, eval_annotation_filepath: str):
    with open(vqa2_annotation_filepath, 'r') as f:
        vqa2_annotation = json.load(f)
    
    # get annotations
    annotations = vqa2_annotation['annotations']
    del vqa2_annotation
    
    # convert
    eval_annotation = {}
    for annotation in tqdm(annotations, desc = 'Convert to needed style'):
        question_id = annotation['question_id']
        eval_annotation[question_id] = annotation
    
    with open(eval_annotation_filepath, 'w') as f:
        json.dump(eval_annotation, f)


