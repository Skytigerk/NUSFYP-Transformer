# 问答系统
from transformers import pipeline
question_answerer = pipeline('question-answering')
sentence = ({
    'question': ' Where can I find the book?',
    'context': 'I think the book is in the library'
})

results = question_answerer(sentence )
print(f"score:{round(results['score'], 4)}, start:{results['start']}, end:{results['end']}, answer:{results['answer']}")
