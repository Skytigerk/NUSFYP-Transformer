# 语义判断
# from transformers import pipeline
# # pipeline.classifier('no!')
# classifier = pipeline('sentiment-analysis')
# sentences = ["what kind of words"]
#              # "I am happy",
#              # "my dad is angry and I am happy"]
# results = classifier(sentences)
#
# for result in results:
#     print(f"label:{result['label']},with score:{round(result['score'], 4)}")

# 问答系统
# from transformers import pipeline
# question_answerer = pipeline('question-answering')
# sentence = ({
#     'question': ' Where can I find the book?',
#     'context': 'I think the book is in the library'
# })
#
# results = question_answerer(sentence )
# print(f"score:{round(results['score'], 4)}, start:{results['start']}, end:{results['end']}, answer:{results['answer']}")

# 图像识别
# from transformers import pipeline
# vision_classifier=pipeline(task="image-classification")
# preds = vision_classifier(
#     images= 'D:/Code/python/lighting_transformer/transformers/examples/pytorch/tiger_learn/run_no_train/elephant1.jpg'
# )
# for pred in preds:
#     print(f"score:{round(pred['score'], 4)}, label: {pred['label']}" )

# 图像问答
from transformers import pipeline
vqa = pipeline(task="vqa")
image= 'D:/Code/python/lighting_transformer/transformers/examples/pytorch/tiger_learn/run_no_train/elephant1.jpg'
question="where is elephant?"
preds=vqa(image=image,question=question)
for pred in preds:
    print(f"score:{round(pred['score'], 4)}, answer: {pred['answer']}" )