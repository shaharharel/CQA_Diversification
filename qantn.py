import tensorflow as tf
import params
from nltk.tokenize import RegexpTokenizer

def inference(batch_placeholders, similarity_placeholder,init_word_embeds,word_to_num,num_to_word):
    print("Begin inference:")
    print("Creating variables")

    E = tf.Variable(init_word_embeds,dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([params.lstm_size,params.lstm_size,params.slice_size],minval=-1.0/params.lstm_size,maxval=1.0/params.lstm_size,name='W'))
    V = tf.Variable(tf.random_uniform([params.slice_size, 2*params.lstm_size],minval=-1.0/(2*params.lstm_size),maxval=1.0/(2*params.lstm_size)))
    b = tf.Variable(tf.zeros([1,params.slice_size]),name='b')
    U = tf.Variable(tf.random_uniform([1,params.slice_size],minval=-1.0/params.slice_size,maxval=1.0/params.slice_size))
    lstm = createLSTM(params.lstm_size)
    print("Calcing sentences2vec")
    question_vec, pos_answer_vec, neg1 , neg2 , neg3 = tf.split(1, params.corrupt_size+2,batch_placeholders)
    #scr_pos_answer, scr_neg1 , scr_neg2 , scr_neg3 = tf.split(1, params.corrupt_size+1,similarity_placeholder)
    #similarity_scores = tf.cast(similarity_placeholder, tf.float32)
    question_vec = tf.squeeze(question_vec)
    pos_answer_vec = tf.squeeze(pos_answer_vec)
    neg1 = tf.squeeze(neg1)
    neg2 = tf.squeeze(neg2)
    neg3 = tf.squeeze(neg3)
    #scr_pos_answer = tf.squeeze(scr_pos_answer)
    #scr_neg1 = tf.squeeze(scr_neg1)
    #scr_neg2 = tf.squeeze(scr_neg2)
    #scr_neg3 = tf.squeeze(scr_neg3)
    #question_vec = tf.reduce_mean(tf.gather(E,question_vec),1)
    question_vec = train_sentence2vectorLSTM(lstm, E, question_vec, False)
    pos_answer_vec = train_sentence2vectorLSTM(lstm, E, pos_answer_vec, True)
    neg1 = train_sentence2vectorLSTM(lstm, E, neg1, True)
    neg2 = train_sentence2vectorLSTM(lstm, E, neg2, True)
    neg3 = train_sentence2vectorLSTM(lstm, E, neg3, True)
    
    #new_p = tf.zeros([pparams.lstm_size+1])
    #pos_answer_vec = tf.reshape(pos_answer_vec, [-1])
    #print scr_pos_answer.get_shape
    #pos_answer_vec = tf.concat(1,[pos_answer_vec,scr_pos_answer])
    #neg1 = tf.concat(1,[neg1,scr_neg1])
    #neg2 = tf.concat(1,[neg2,scr_neg2])
    #neg3 = tf.concat(1,[neg3,scr_neg3])
    #pos_answer_vec = tf.reduce_mean(tf.gather(E, pos_answer_vec), 1)
    #neg1 = tf.reduce_mean(tf.gather(E, neg1), 1)
    #neg2 = tf.reduce_mean(tf.gather(E, neg2), 1)
    #neg3 = tf.reduce_mean(tf.gather(E, neg3), 1)


    tensors=[]
    for i in range(params.slice_size):
        tensor = tf.reduce_sum(pos_answer_vec*tf.matmul(question_vec,W[:, :,i]),1)
        tensors.append(tensor)

    score_pos = tf.pack(tensors)
    vec_concat = tf.transpose(tf.matmul(V, tf.transpose(tf.concat(1, [question_vec, pos_answer_vec]))))
    score_pos = tf.matmul(tf.nn.relu(tf.transpose(score_pos) + vec_concat + b), tf.transpose(U))

    negative=[]
    for i in [neg1,neg2,neg3]:
        tensors = []
        for j in range(params.slice_size):
            tensor = tf.reduce_sum(i * tf.matmul(question_vec, W[:, :, j]), 1)
            tensors.append(tensor)

        score_neg = tf.pack(tensors)
        vec_concat = tf.transpose(tf.matmul(V, tf.transpose(tf.concat(1, [question_vec, i]))))
        score_neg = tf.matmul(tf.nn.relu(tf.transpose(score_neg) + vec_concat + b), tf.transpose(U))
        negative.append(score_neg)

    return [score_pos,negative[0],negative[1],negative[2]]

def inference_eval(batch_placeholders, similarity_placeholder, init_word_embeds,word_to_num,num_to_word):
    print("Begin inference:")
    print("Creating variables")
    E = tf.Variable(init_word_embeds,dtype=tf.float32)
    W = tf.Variable(tf.truncated_normal([params.wv_size,params.wv_size,params.slice_size]),name='W')
    V = tf.Variable(tf.zeros([params.slice_size, 2*params.wv_size]))
    b = tf.Variable(tf.zeros([1,params.slice_size]),name='b')
    U = tf.Variable(tf.zeros([1,params.slice_size]))
    #lstm = createLSTM(300)
    print("Calcing sentences2vec")
    question_vec, answer_vec = tf.split(1, 2, batch_placeholders)
    similarity_scores = tf.cast(similarity_placeholder, tf.float32)
    question_vec = tf.squeeze(question_vec)
    answer_vec = tf.squeeze(answer_vec)
    question_vec = tf.reduce_mean(tf.gather(E, question_vec),1)
    #question_vec = train_sentence2vectorLSTM(lstm, E, question_vec, False)
    #pos_answer_vec = train_sentence2vectorLSTM(lstm, E, pos_answer_vec, True)
    #neg1 = train_sentence2vectorLSTM(lstm, E, neg1, True)
    #neg2 = train_sentence2vectorLSTM(lstm, E, neg2, True)
    #neg3 = train_sentence2vectorLSTM(lstm, E, neg3, True)
    answer_vec = tf.reduce_mean(tf.gather(E, answer_vec), 1)
    tensors=[]
    for i in range(params.slice_size):
        tensor = tf.reduce_sum(answer_vec*tf.matmul(question_vec,W[:, :,i]),1)
        tensors.append(tensor)
    score_pos = tf.pack(tensors)
    vec_concat = tf.transpose(tf.matmul(V, tf.transpose(tf.concat(1, [question_vec, answer_vec]))))
    score_pos = tf.matmul(tf.nn.relu(tf.transpose(score_pos) + vec_concat + b), tf.transpose(U))
    return score_pos

def loss(predictions):

    print("Beginning computing loss")
    temp = tf.maximum(0.0,tf.sub(predictions[1],predictions[0])+1)
    temp += tf.maximum(0.0,tf.sub(predictions[2],predictions[0]) + 1)
    temp += tf.maximum(0.0,tf.sub(predictions[3],predictions[0]) + 1)
    #temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp)
    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    loss = temp1 + (params.regularization * temp2)
    return loss

def training(loss, learningRate):
    print("Begin training")
    return tf.train.AdamOptimizer(learningRate).minimize(loss)

def eval(predictions):

    return

def createLSTM(hidden_dim):
    return tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)

def train_sentence2vectorLSTM(lstm, wv ,sentence, flag_lstm):
    count = 0
    state = tf.ones([params.batch_size, lstm.state_size])
    with tf.variable_scope("myrnn") as scope:
        for i in range(params.max_sentence_length):
            if count > 0 or flag_lstm is True:
                scope.reuse_variables()
            output, state = lstm(tf.reshape(tf.gather(wv,sentence[:,i]),
                                            shape=(params.batch_size, params.wv_size)), state)
            count += 1
    return output


