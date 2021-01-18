import pandas as pd
from pycorenlp import StanfordCoreNLP
import glob

class isQuestionBasic():

    # Init Constructor
    # Initialize StanfordCore NLP local instance on port 9000
    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9000')

    # Input: Sentence to be predicted
    # Processing: 1. Uses Stanfors NLP's 'annotate' method to create Parse Tree
    # 2. Checks for occurence of 'SQ' or 'SBARQ' in the parse tree
    # Return: 1 - If sentence is question | 0 - If sentence is not a question
    def isQuestion(self, sentence):
        if '?' in sentence:
            return 1
        output = self.nlp.annotate(sentence, properties={
            'annotators': 'parse',
            'outputFormat': 'json',
            'timeout': 1000,
        })
        try:
            if ('SQ' or 'SBARQ') in output['sentences'][0]["parse"]:
                return 1
            else:
                return 0
        except:
            print("sentence too long")


for file in glob.glob("transcripts/KAM/*.txt"):
    filename = file.split("\\")[-1].split(".")[0]
    isQuestionBasic_obj = isQuestionBasic()
    df = pd.read_csv(file, sep='\t')
    df['is_question'] = df['QUERY'].apply(isQuestionBasic_obj.isQuestion)
    df.to_csv('output/'+filename+'.csv', index=False)

