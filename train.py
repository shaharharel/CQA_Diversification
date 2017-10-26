import tensorflow as tf
import numpy as np
import datetime
import qantn
import params
import random
from load_data import *
import evaluation



def get_next_batch(batch_size,corrupt_size,questions):
    rand = random.sample(xrange(len(questions)), batch_size)
    batch=[]
    bm25_sims = []
    for i in rand:
        point=[]
        sim_point = []
        point.append(questions[i].question)
        aspects = [len(ans.aspects) for ans in questions[i].answers]
        lst = np.argsort(aspects)
        a1 = np.random.choice([lst[-1],lst[-2]])
        point.append(questions[i].answers[a1].answer)
        sim_point.append(questions[i].answers[a1].score)
        worst = np.random.choice(lst[:5],3)
        point += [questions[i].answers[ind].answer for ind in worst]
        sim_point += [questions[i].answers[ind].score for ind in worst]
        batch.append(point)
        bm25_sims.append(sim_point)
        #print bm25_sims
        #exit()
    return np.array(batch), bm25_sims

def get_next_batch_broken(batch_size,corrupt_size,questions):
    rand = random.sample(xrange(len(questions)), batch_size)
    batch=[]
    for i in rand:
        point=[]
        point.append(questions[i].question)
        aspects = [len(ans.aspects) for ans in questions[i].answers]
        lst = np.argsort(aspects)
        positive = []
        for ind in lst:
            if aspects[ind] >= 4:
                positive.append(ind)
        if len(positive)<1:
             positive.append(lst[-1])
        a1 = np.random.choice(positive)
        point.append(questions[i].answers[a1].answer)
        negative = []
        for ind in lst:
            if aspects[ind] == 0 or aspects[ind] == 1:
                negative.append(ind)
        while len(negative)<3:
             negative.append(np.random.choice([lst[0],lst[1]]))
             print [aspects[j] for j in negative]
             print aspects
        worst = np.random.choice(negative,3)
        point += [questions[i].answers[ind].answer for ind in worst]
        batch.append(point)
    return np.array(batch)

def get_batch_for_eval(batch_size,questions,answers):
    batch=[]
    rand = random.sample(xrange(len(questions)), batch_size)
    for i in rand:
        for j in answers[i+1]:
            if len(batch)==params.batch_size:
                return batch , rand, i
            point = []
            point.append(questions[i])
            point.append(j)
            point.append(j)
            point.append(j)
            point.append(j)
            batch.append(point)

def run_training(data, fold):
    print("Begin!")
    print("Load training data...")
    wv, word_to_num, num_to_word = load_wv('vocabReal.txt', 'vectorsReal.txt')
    #print "run training on num of questions = " + str(len(questions))
    with tf.Graph().as_default():
        print("Starting to build graph " + str(datetime.datetime.now()))
        batch_placeholders = tf.placeholder(tf.int32, shape=(None,params.corrupt_size+2,params.max_sentence_length ))
        similarity_placeholders = tf.placeholder(tf.float32, shape=(None, params.corrupt_size+1, params.max_sentence_length))
        inference = qantn.inference(batch_placeholders, similarity_placeholders,wv,word_to_num,num_to_word)
        loss = qantn.loss(inference)
        training = qantn.training(loss, params.learning_rate)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=1000)
        for i in range(1, params.num_iters):
            print("Starting iter " + str(i) + " " + str(datetime.datetime.now()))
            data_batch,bm25 = get_next_china_batch(params.batch_size, params.corrupt_size, data)
            data_batch,_ = fill_wv_from_batch(data_batch, word_to_num)
            similarities = np.zeros([params.batch_size,params.corrupt_size+1])
            feed_dict = {batch_placeholders: data_batch,similarity_placeholders:bm25}
            _, loss_value = sess.run([training, loss], feed_dict=feed_dict)
            if i % params.save_per_iter == 0:
              saver.save(sess, 'bm25_omari_china' + str(fold) + '/' + str(i) + '.sess')
            print loss_value

def initialize():
    QA = load_json_file('really_train')
    QA += load_json_file('really_test')
    questions, qnswers = parse_xml_Data(QA, False, True, None)
    create_word_vectors_files()

if __name__ == '__main__':
    data = omari_data_load()
    run_training(data, fold)




