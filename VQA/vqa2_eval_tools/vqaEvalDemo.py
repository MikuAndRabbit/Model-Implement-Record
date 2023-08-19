from vqaEval import VQAEval
import matplotlib.pyplot as plt
import json


"""
need provided:
	model_res_dict_filepath: the filepath of predicted result of model
	label_dict_filepath: the filepath of label file
saved filepath: accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile, result_image_path
"""
model_res_dict_filepath = r''
label_dict_filepath = r''
accuracyFile = r''
evalQAFile = r''
evalQuesTypeFile = r''
evalAnsTypeFile = r''
result_image_path = r''


# load label dict & model predict
with open(label_dict_filepath, 'r') as f:
	_label_dict = json.load(f)
label_dict = {int(k): v for k, v in _label_dict.items()}
del _label_dict

with open(model_res_dict_filepath, 'r') as f:
	_res_dict = json.load(f)
res_dict = {int(question_id): predict_answer for question_id, predict_answer in _res_dict.items()}
del _res_dict

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(res_dict = res_dict, label_dict = label_dict, digits_num = 2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate() 

# print accuracies
print("")
print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
print("Per Question Type Accuracy is the following:")
for quesType in vqaEval.accuracy['perQuestionType']:
	print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
print("")
print("Per Answer Type Accuracy is the following:")
for ansType in vqaEval.accuracy['perAnswerType']:
	print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
print("")
 
# save evaluation results to certain path
json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

# plot accuracy for various question types
plt.figure(figsize=(45, 15))
plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation=45, fontsize=10, ha='right')
plt.title('Per Question Type Accuracy', fontsize=10)
plt.xlabel('Question Types', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.savefig(result_image_path)
plt.show()
