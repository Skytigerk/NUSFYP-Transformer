# 语义判断
from transformers import pipeline
# pipeline.classifier('no!')
classifier = pipeline('sentiment-analysis')
sentences = ["what kind of words"]
             # "I am happy",
             # "my dad is angry and I am happy"]
results = classifier(sentences)
for result in results:
    print(f"label:{result['label']},with score:{round(result['score'], 4)}")