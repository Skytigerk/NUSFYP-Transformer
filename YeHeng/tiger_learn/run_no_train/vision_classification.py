# 图像识别
from transformers import pipeline
vision_classifier=pipeline(task="image-classification")
preds = vision_classifier(
    images= './elephant1.jpg'
)
for pred in preds:
    print(f"score:{round(pred['score'], 4)}, label: {pred['label']}" )