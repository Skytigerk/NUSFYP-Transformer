
# 图像问答
from transformers import pipeline
vqa = pipeline(task="vqa")
image= 'D:/Code/python/lighting_transformer/transformers/examples/pytorch/tiger_learn/run_no_train/elephant1.jpg'
question="where is elephant?"
preds=vqa(image=image,question=question)
for pred in preds:
    print(f"score:{round(pred['score'], 4)}, answer: {pred['answer']}" )