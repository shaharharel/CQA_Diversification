import json
import numpy as np
from gensim.models import word2vec
from nltk.tokenize import RegexpTokenizer
import params
import random
from stop_words import get_stop_words
import evaluation

class Answer:
    def ominit(self, answerObject, aspectObject):
        self.answer = " ".join(answerObject['answer_text'].split())
        self.is_best_answer = answerObject['is_best_answer']
        self.has_source = False
        if 'answer_sources' in answerObject:
            self.has_source = True
        self.thumbs_up = answerObject['thumbs_down']
        self.thumbs_down =  answerObject['thumbs_down']
        self.aspects_max_size = len(aspectObject['proposition_clusters'])
        self.aspects=[]
        self.aspects_importance = []
        self.score = 0
        i = 0
        for aspect in aspectObject['proposition_clusters']:
            for example in aspect:
                if example.lower() in self.answer.lower():
                   self.aspects.append(i)
                   break
                if self.is_aspect_in_answer(example, self.answer):
                   if example in ['no. ','NO!']:
                        continue
                   self.aspects.append(i)
                   break
            i+=1
        self.aspects = list(set(self.aspects))
    def __init__(self, answer, aspects):
        self.answer = answer
        self.aspects_max_size = len(aspects)
        self.aspects = aspects
        self.score = 0
        self.aspects_importance = []
    def is_aspect_in_answer(self, aspect, answer):
       tokenizer = RegexpTokenizer(r'\w+')
       a1 = tokenizer.tokenize(aspect.lower())
       a2 = tokenizer.tokenize(answer.lower())
       equal = len(set(a1).intersection(set(a2))) == len(a1)
       if equal:
          indexs=[]
          count=0
          for i in a1:
              indexs.append(a2.index(i))
          for i in range(len(indexs)-1):
              count += (abs(indexs[i+1]-indexs[i])-1)
          if count>5:
              equal=False
       else:
          if len(a1)>7 and len(a2)<200 and len(set(a1).intersection(set(a2))) >= (float(len(a1)-1)/len(a1))*len(a1):
              equal=True
              #print aspect
              #print answer
              #print '---------------------------------------------'
       return equal

class Question:
    def ominit(self, questionObject, aspectObject):
        str = ''
        if 'body' in questionObject:
            str = " ".join(questionObject['body'].split())
        self.question = " ".join(questionObject['title'].split()) + " " + str
        self.id = questionObject['question_id']
        self.answers = []
        self.matrix_scores = []
        self.aspectsSize = len(aspectObject['proposition_clusters'])
        self.aspects_importance = {k:0 for k in range(self.aspectsSize)}
        i = 0
        for aspect in aspectObject['proposition_clusters']:
            self.aspects_importance[i] = len(aspect)
            i+=1
        for answer in questionObject['answers']:
             a2 = Answer([],[])
             a2.ominit(answer, aspectObject)
             self.answers.append(a2)
    def __init__(self, id, text, aspectSize):
        self.question = text
        self.id = id
        self.answers = []
        self.matrix_scores = []
        self.aspectsSize = aspectSize
        self.aspects_importance = {k: 0 for k in range(self.aspectsSize)}
def load_json_file(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data

def load_new_data():
    with open('liveQAData', 'r') as f:
        good_q = json.load(f)
    with open('HitIdToAnswer', 'r') as f:
        id_to_answers = json.load(f)
    qs = []
    id = 0
    for q in good_q:
        aspects = set()
        for hidID in good_q[q]:
            aspects = aspects.union(set(good_q[q][hidID]))
        new_q = Question(id, q, len(aspects))
        for hidID in good_q[q]:
            ans = Answer(id_to_answers[hidID], list(set(good_q[q][hidID])))
            for aspect in ans.aspects:
                new_q.aspects_importance[aspect] += 1
            ans.aspects_max_size = len(aspects)
            new_q.answers.append(ans)
        qs.append(new_q)
        id += 1
    return qs

def fill_wv_from_batch(batch,word_to_num):
    new_batch = []
    tokenizer = RegexpTokenizer(r'\w+')
    lengths_vector = [[],[],[],[],[],[]]
    stop = get_stop_words('en')
    #import symspell_python
    #symspell_python.init()
    for point in batch:
        poi = []
        for ind,sentence in enumerate(point):
            raw = sentence.lower()
            tokens = tokenizer.tokenize(raw)
            temp = []
            for i in tokens:
                if i in stop:
                   continue
                try:
                    temp.append(word_to_num[i])
                except:
                    try:
                        tk=symspell_python.suggest(i)
                        temp.append(word_to_num[tk])
                    except:
                        continue
            temp = temp[:params.max_sentence_length]
            length = len(temp)
            incd = False
            if length == 0:
               length+=1
               incd=True
            lengths_vector[ind].append(length)
            if incd is True:
                #print "dec"
                length-=1
            if length > params.max_sentence_length:
                print("take a look - bug")
                break
            if length < params.max_sentence_length:
                for i in range(params.max_sentence_length - length):
                    temp.append(word_to_num['AAABBBCCC'])
            poi.append(np.array(temp))
        new_batch.append(poi)
    return new_batch,lengths_vector

def parse_xml_Data(QA, id=False, saveVocab=False, word_to_num=None):
    questions = []
    answers = [[]]
    if not id:
        for question in QA:
            questions.append(question['content'])
            ans = []
            for an in question['nbestanswers']:
                ans.append(an)
            answers.append(ans)
    else:
        fill_wv_from_questions(QA, word_to_num, questions, answers)
    if saveVocab:
        vocab = set()
        # loop through document list
        tokenizer = RegexpTokenizer(r'\w+')
        for i in questions:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            for tok in tokens:
                vocab.add(tok)
        for ans in answers:
            for i in ans:
                raw = i.lower()
                tokens = tokenizer.tokenize(raw)
                for tok in tokens:
                    vocab.add(tok)
        f = open('vocab.txt', 'w')
        for i in vocab:
            s = (i + '\n').encode('utf-8').strip()
            f.write(s + '\n')
        f.close()
    return questions, answers

def fill_wv_from_questions(QA, word_to_num, questions, answers):
    tokenizer = RegexpTokenizer(r'\w+')
    stop = get_stop_words('en')
    count_q=0
    count_a=0
    vectorless = set()
    withvector = set()
    import symspell_python
    symspell_python.init()
    for question in QA:
        for field in ['content', 'nbestanswers']:
            if field == 'content':
                raw = question[field].lower()
                tokens = tokenizer.tokenize(raw)
                temp = []
                for i in tokens:
                    if i in stop:
                       continue
                    try:
                        temp.append(word_to_num[i])
                        withvector.add(i)
                    except:
                       tk=symspell_python.suggest(i)
                       try:
                           temp.append(word_to_num[tk])
                       except:
                           vectorless.add(tk)
                       continue
                if len(temp) > params.max_sentence_length:
                    count_q+=1
                temp = temp[:params.max_sentence_length]
                length = len(temp)
                if length > params.max_sentence_length:
                    print("take a look - bug")
                    break
                if length < params.max_sentence_length:
                    for i in range(params.max_sentence_length - length):
                        temp.append(word_to_num['AAABBBCCC'])
                questions.append(np.array(temp))
                ans = []
            if field == 'nbestanswers':
                for an in question[field]:
                    raw = an.lower()
                    tokens = tokenizer.tokenize(raw)
                    temp = []
                    for i in tokens:
                        if i in stop:
                           continue
                        try:
                           temp.append(word_to_num[i])
                           withvector.add(i)
                        except:
                           tk=symspell_python.suggest(i)
                           try:
                              temp.append(word_to_num[tk])
                           except:
                              vectorless.add(tk)
                    if len(temp) > params.max_sentence_length:
                          count_a+=1
                    temp = temp[:params.max_sentence_length]
                    length = len(temp)
                    if length > params.max_sentence_length:
                        print("bug in nbestanswer")
                        break
                    if length < params.max_sentence_length:
                        for i in range(params.max_sentence_length - length):
                            temp.append(word_to_num['AAABBBCCC'])
                    ans.append(np.array(temp))
                answers.append(np.array(ans))
    print "number of bad questions"+str(count_q)
    print "number of answers "+str(count_a)
    print "word with no vectors"+str(len(vectorless))
    print "word with vectors"+str(len(withvector))
    #print vectorless
    return None

def create_word_vectors_files(vocab=False):
    model = word2vec.Word2Vec.load('vectors_100')
    f = open('vocabReal.txt', 'w')
    g = open('vectorsReal.txt', 'w')
    for i in open('vocab.txt'):
        try:
            g.write(" ".join(map(str, model[i.strip()])) + '\n')
            f.write(i.strip() + '\n')
        except:
            continue
    f.close()
    g.close()

def get_diversed(question, answer,i):
    lst = []
    for ans in question.answers:
        if ans == answer:
           continue
        lst.append((ans,len(set(ans.aspects) & set(answer.aspects))))
    lst.sort(key=lambda x : x[1])
    return lst[i][0]
def get_diversed_lda(model, dictionary, question, answer ,i):
    lst = []
    for ans in question.answers:
        if ans == answer:
           continue
        lst.append((ans,evaluation.similarity(model, dictionary, ans.answer, answer.answer)))
    lst.sort(key=lambda x : x[1])
    return lst[i][0]
def omari_batch(model, dictionary):
    aspects_file=open('aspects').read()
    data1 = json.loads(aspects_file)
    questions_file = open('questions').read()
    data2 = json.loads(questions_file)
    id_to_aspects = {v['question_id']:v for v in data1}
    questions = []
    for q in data2:
        questions.append(Question(q, id_to_aspects[q['question_id']]))
    batch = []
    questions = questions[:88]
    print len(questions)
    for q in questions:
        point = [q.question]
        best = None
        for ans in q.answers:
            if ans.is_best_answer is True:
                best = ans
                break
        if best is None:
            print "errrorrR"
            continue
        point.append(best.answer)
        point.append(get_diversed_lda(model, dictionary, q,best,0).answer)
        point.append(get_diversed_lda(model, dictionary, q,best,-1).answer)
        randq = random.sample(questions,1)[0]
        randb = random.sample(randq.answers,1)[0]
        point.append(randb.answer)
        point.append(get_diversed_lda(model, dictionary, randq,randb,-1).answer)
        batch.append(point)
    print len(batch)
    return batch+random.sample(batch, params.batch_size - len(batch))
        

def omari_data_load():
    aspects_file=open('aspects').read()
    data1 = json.loads(aspects_file)
    questions_file = open('questions').read()
    data2 = json.loads(questions_file)
    id_to_aspects = {v['question_id']:v for v in data1}
    questions = []
    for q in data2:
        q2 = Question([],[],0)
        q2.ominit(q, id_to_aspects[q['question_id']])
        questions.append(q2)
    return questions

def split_omari_data():
    questions_file = open('questions').read()
    data2 = json.loads(questions_file)
    f = open('test_set', 'w')
    json.dump(data2[:54], f)
    f.close()
    f = open('validation_set', 'w')
    json.dump(data2[55:], f)
    f.close()
    print len(data2)

def load_params(data):
    metrics_file = open('metrics').read()
    metrics = json.loads(metrics_file)
    dict = {q.question_id:q for q in data}
    for id in metrics:
        lst = dict[id]
        dict[id].err_ia = lst[0]
        dict[id].ndcg = lst[1:5]
        dict[id].support = lst[6:15]
        dict[id].novelty = lst[16:]

def sample_corrupted_2(questions,answers):
    q_lst = random.sample(xrange(len(questions)),params.corrupt_size)
    corrupt = []
    for q in q_lst:
        corrupt.append(answers[q+1][random.sample(xrange(len(answers[q+1])),1)[0]])
    return corrupt

def get_test_data(questions, answers):
    batch = []
    count = 0
    last_q = 0
    count2 = 0
    for i in xrange(len(questions)):
        point = []
        point.append(questions[i])
        point.append(answers[i+1][0])
        point += sample_corrupted_2(questions, answers)
        batch.append(point)
    return np.array(batch)

def invert_dict(d):
    return {v: k for k, v in d.iteritems()}

def load_wv(vocabfile, wvfile):
    wv = np.loadtxt(wvfile, dtype=float)
    with open(vocabfile) as fd:
        words = [line.strip() for line in fd]
    words.append('AAABBBCCC')
    num_to_word = dict(enumerate(words))
    word_to_num = invert_dict(num_to_word)
    temp = np.zeros((wv[0].shape))
    temp.shape = (1, temp.shape[0])
    wv = np.concatenate((wv, temp))
    return wv, word_to_num, num_to_word






