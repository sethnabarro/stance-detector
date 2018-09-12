# Stance Detection

The aim of this project is to build a model to infer the author's stance on a subject, given a tweet. For more background see SemEval 2016 Task 6 (http://alt.qcri.org/semeval2016/task6/). 

### Requirements

Code designed for Python 3.5.2. All required libraries can be installed by navigating into the directory and running

```
pip install -r requirements.txt
```

(assuming pip installed). Train and test sets (with gold labels - see http://alt.qcri.org/semeval2016/task6/index.php?id=data-and-tools) are assumed to reside in a 'data/' folder within the repo.

### Run the Code

```
python main.py
```

will print results on model performance to console and generate corresponding .tex tables.
