import tensorflow as tf
import re
import string

BATCH_SIZE = 50
MAX_WORDS_IN_REVIEW = 40  # Maximum length of a review to consider
EMBEDDING_SIZE = 50	 # Dimensions for each word vector
LSTM_SIZE = 128
FC_UNITS = 256
NUM_LAYERS = 2
NUM_CLASSES = 2
LEARNING_RATE = 0.0001

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we', 'its',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',
                  'you', 'herself', 'has', 'just', 'where', 'too', 'only',
                  'myself', 'which', 'those', 'i', 'after', 'few', 'whom',
                  't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by',
                  'doing', 'it', 'how', 'further', 'was', 'here', 'than','wouldn', 'won', 'weren', 'wasn', 'shouldn', 'shan', 'needn',
                  'mustn', 'mightn', 'ma', 'isn', 'haven', 'hasn', 'hadn', 'doesn', 'didn', 'couldn', 'aren', 'ain', 'ain', 'y'})

stop_words2 = {'over', 'shouldn', 'before', 'd', 'had', 'those', 'yourself', 'whom', 'you', 'few', 'mustn', 'to',
               'where', 'of', 'she', 'did', 'isn', 'am', 're', 'haven', 'were', 'doesn', 'yours', 'my', 'so', 'above',
               'out', 'that', 'more', 'him', 'why', 'here', 've', 'hadn', 'I', 'couldn', 'itself', 'this', 'ourselves',
               'does', 'but', 'having', 'about', 'yourselves', 'been', 'because', 'such', 'wouldn', 'them', 'through',
               'from', 'no', 'both', 'who', 'what', 'doing', 'very', 'their', 'into', 'further', 'hers', 'y', 'has',
               'won', 'own', 'if', 'a', 'mightn', 'was', 'not', 'should', 'are', 'himself', 'any', 'each', 'same',
               'wasn', 'o', 'other', 's', 'we', 'your', 'now', 'he', 'be', 'how', 'after', 'have', 'being', 'didn',
               'some', 'it', 'its', 'hasn', 'for', 'too', 'with', 'which', 'needn', 'until', 'there', 'll', 'between',
               'most', 'theirs', 'myself', 't', 'themselves', 'then', 'in', 'our', 'just', 'under', 'and', 'aren',
               'herself', 'up', 'an', 'these', 'do', 'her', 'shan', 'as', 'i', 'off', 'when', 'at', 'm', 'they', 'than',
               'on', 'once', 'will', 'only', 'ain', 'again', 'or', 'ma', 'can', 'his', 'the', 'against', 'is', 'weren',
               'below', 'me', 'nor', 'by', 'all', 'during', 'while', 'ours', 'down'}

stop_words = stop_words.union(stop_words2)
# other_punctuation = {'<br />'}

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here
    that is manipulation at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """

    review = decontracted(review.lower())
    processed_review = [word for word in review.lower().translate(
                                            str.maketrans('', '',
                                            string.punctuation)).split() if word not in stop_words]
    return processed_review


def define_graph():
    """
    Implement your model here. You will need to define placeholders,
    for the input and labels. Note that the input is not strings of
    words, but the strings after the embedding lookup has been applied
    (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py.
    You should read this file and ensure your code here is compatible.

    Consult the assignment specification for details of which
    parts of the TF API are permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    #with tf.name_scope("input_data"):
    # with tf.name_scope("input_data"):
    # tf.reset_default_graph()
    #input_data = tf.placeholder(tf.float32,[None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],name='input_data')
    input_data = tf.placeholder(tf.float32, [None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')

    # with tf.name_scope("labels"):
    labels = tf.placeholder(tf.int32,[None, NUM_CLASSES],name='labels')

    # with tf.name_scope("dropout_keep_prob"):
    dropout_keep_prob = tf.placeholder_with_default(0.5,shape=(),name='dropout_keep_prob')

    # with tf.name_scope("RNN_LAYER"):s
    lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    drop = tf.contrib.rnn.DropoutWrapper(cell=lstm,output_keep_prob=dropout_keep_prob)

    initial_state = drop.zero_state(BATCH_SIZE, tf.float32)

    # with tf.name_scope("RNN_DYNAMIC_CELL"):
    outputs, final_state = tf.nn.dynamic_rnn(drop,input_data,initial_state=initial_state,dtype=tf.float32)

    # with tf.name_scope("fully_connected"):
    weights = tf.truncated_normal_initializer(stddev=0.1)

    biases = tf.zeros_initializer()

    preds = tf.contrib.layers.fully_connected(
        outputs[:, -1],
        num_outputs=2,
        activation_fn=tf.nn.softmax,
        weights_initializer=weights,
        biases_initializer=biases)

    preds = tf.contrib.layers.dropout(preds, dropout_keep_prob)

    # with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=labels),name='loss')

    # with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # with tf.name_scope("accuracy"):
    #accuracy = tf.get_variable(name="accuracy", shape=1)
    #accuracy = tf.contrib.metrics.accuracy(tf.cast(tf.round(preds), dtype=tf.int32),labels,name='accuracy')
    accuracy_1 = tf.contrib.metrics.accuracy(tf.cast(tf.round(preds), dtype=tf.int32),labels,name='accuracy_1')
    accuracy = tf.identity(accuracy_1, name='accuracy')
    # accuracy = tf.reduce_mean(tf.cast(preds,tf.float32),name='accuracy')
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(preds)), labels), tf.float32), name='accuracy')
    # print(accuracy.name)
    # assert accuracy.name == "accuracy:0"
    # assert (input_data.name, labels.name, accuracy.name, loss.name) == (
    # "input_data:0", "labels:0", "accuracy:0", "loss:0"), 'Incorrect name. Got {}.'.format(
    #     (input_data.name, labels.name, accuracy.name, loss.name))

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
