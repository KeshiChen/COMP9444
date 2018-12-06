import tensorflow as tf
import re
import string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 40  # Maximum length of a review to consider
EMBEDDING_SIZE = 50	 # Dimensions for each word vector
LSTM_SIZE = 128
FC_UNITS = 256
NUM_LAYERS = 2
NUM_CLASSES = 2
LEARNING_RATE = 0.001

# pos+neg相关性分析，把中性词加入stopword
# 长度不够，用stop word替换
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

# stop_words2 = {'over', 'shouldn', 'before', 'd', 'had', 'those', 'yourself', 'whom', 'you', 'few', 'mustn', 'to',
#                'where', 'of', 'she', 'did', 'isn', 'am', 're', 'haven', 'were', 'doesn', 'yours', 'my', 'so', 'above',
#                'out', 'that', 'more', 'him', 'why', 'here', 've', 'hadn', 'I', 'couldn', 'itself', 'this', 'ourselves',
#                'does', 'but', 'having', 'about', 'yourselves', 'been', 'because', 'such', 'wouldn', 'them', 'through',
#                'from', 'no', 'both', 'who', 'what', 'doing', 'very', 'their', 'into', 'further', 'hers', 'y', 'has',
#                'won', 'own', 'if', 'a', 'mightn', 'was', 'not', 'should', 'are', 'himself', 'any', 'each', 'same',
#                'wasn', 'o', 'other', 's', 'we', 'your', 'now', 'he', 'be', 'how', 'after', 'have', 'being', 'didn',
#                'some', 'it', 'its', 'hasn', 'for', 'too', 'with', 'which', 'needn', 'until', 'there', 'll', 'between',
#                'most', 'theirs', 'myself', 't', 'themselves', 'then', 'in', 'our', 'just', 'under', 'and', 'aren',
#                'herself', 'up', 'an', 'these', 'do', 'her', 'shan', 'as', 'i', 'off', 'when', 'at', 'm', 'they', 'than',
#                'on', 'once', 'will', 'only', 'ain', 'again', 'or', 'ma', 'can', 'his', 'the', 'against', 'is', 'weren',
#                'below', 'me', 'nor', 'by', 'all', 'during', 'while', 'ours', 'down'}
stop_words2 = {'secondly', 'all', 'consider', 'pointing', 'whoever', 'results', 'felt', 'four',
               'edu', 'go', 'oldest', 'causes', 'poorly', 'whose', 'certainly', 'biol',
               'everywhere', 'vs', 'young', 'containing', 'presents', 'to', 'does', 'present',
               'th', 'under', 'sorry', 'include', "a's", 'sent', 'insofar', 'consequently',
               'far', 'none', 'every', 'yourselves', 'associated', "we'll", 'immediately',
               'presented', 'did', 'turns', 'having', "they've", 'large', 'p', 'small',
               'thereupon', 'noted', "it'll", "i'll", 'parted', 'smaller', 'says', "you'd",
               'd', 'past', 'likely', 'invention', 'zz', 'zt', 'further', 'even', 'index',
               'what', 'appear', 'giving', 'section', 'brief', 'fifth', 'goes', 'sup', 'new',
               'seemed', 'ever', 'full', "c'mon", 'respectively', 'men', 'here', 'youngest',
               'let', 'groups', 'others', "hadn't", 'along', "aren't", 'obtained', 'great',
               'ref', 'k', 'allows', 'proud', "i'd", 'resulting', 'arent', 'usually', 'que',
               "i'm", 'changes', 'thats', 'hither', 'via', 'followed', 'members', 'merely',
               'ts', 'ask', 'ninety', 'vols', 'viz', 'ord', 'readily', 'everybody', 'use',
               'from', 'working', 'contains', 'two', 'next', 'few', 'therefore', 'taken',
               'themselves', 'thru', 'until', 'today', 'more', 'knows', 'clearly', 'becomes',
               'hereby', 'herein', 'downing', "ain't", 'particular', 'known', "who'll", 'cases',
               'given', 'must', 'me', 'states', 'mg', 'room', 'f', 'this', 'ml', 'when',
               'anywhere', 'nine', 'can', 'mr', 'following', 'making', 'my', 'example',
               'something', 'indicated', 'give', "didn't", 'near', 'high', 'indicates',
               'numbers', 'want', 'arise', 'longest', 'information', 'needs', 'end', 'thing',
               'rather', 'ie', 'get', 'how', 'instead', "doesn't", 'okay', 'tried', 'may',
               'overall', 'after', 'eighty', 'them', 'tries', 'ff', 'date', 'such', 'man',
               'a', 'thered', 'third', 'whenever', 'maybe', 'appreciate', 'q', 'cannot',
               'so', 'specifying', 'allow', 'keeps', 'looking', 'order', "that's", 'six',
               'help', "don't", 'furthering', 'indeed', 'itd', 'mainly', 'soon', 'years',
               'course', 'through', 'looks', 'still', 'its', 'before', 'beside', 'group',
               'thank', "he's", 'selves', 'inward', 'fix', 'actually', 'better', 'willing',
               'differently', 'thanx', 'somethan', 'ours', "'re", 'might', "haven't", 'then',
               'non', 'good', 'affected', 'greater', 'thereby', 'downs', 'auth', "you've",
               'they', 'not', 'now', 'discuss', 'nor', 'nos', 'down', 'gets', 'hereafter',
               'always', 'reasonably', 'whither', 'l', 'sufficiently', 'each', 'found', 'went',
               'higher', 'side', "isn't", 'mean', 'everyone', 'significantly', 'doing', 'ed',
               'eg', 'related', 'owing', 'ex', 'year', 'substantially', 'et', 'beyond',
               "c's", 'puts', 'out', 'try', 'shown', 'opened', 'miss', 'furthermore',
               'since', 'research', 'rd', 're', 'seriously', "shouldn't", "they'll", 'got',
               'forth', 'shows', 'turning', 'state', 'million', 'little', 'quite', "what'll",
               'whereupon', 'besides', 'put', 'anyhow', 'wanted', 'beginning', 'g', 'could',
               'needing', 'hereupon', 'keep', 'turn', 'place', 'w', 'ltd', 'hence', 'onto',
               'think', 'first', 'already', 'seeming', 'omitted', 'thereafter', 'number',
               'thereof', 'yourself', 'done', 'least', 'another', 'open', 'awfully', "you're",
               'differ', 'necessarily', 'indicate', 'ordering', 'inasmuch', 'approximately',
               'anyone', 'needed', 'too', 'hundred', 'gives', 'interests', 'mostly', 'that',
               'exactly', 'took', 'immediate', 'part', 'somewhat', "that'll", 'believe',
               'herself', 'than', "here's", 'begins', 'kind', 'b', 'unfortunately', 'showed',
               'accordance', 'gotten', 'largely', 'second', 'i', 'r', 'were', 'toward',
               'anyways', 'and', 'sees', 'ran', 'thoughh', 'turned', 'anybody', 'say',
               'unlikely', 'have', 'need', 'seen', 'seem', 'saw', 'any', 'relatively',
               'smallest', 'zero', 'thoroughly', 'latter', "i've", 'downwards', 'aside',
               'thorough', 'predominantly', 'also', 'take', 'which', 'wanting', 'greatest',
               'begin', 'added', 'unless', 'shall', 'knew', 'wells', "where's", 'most',
               'eight', 'amongst', 'significant', 'nothing', 'pages', 'parting', 'sub',
               'cause', 'kg', 'especially', 'nobody', 'clear', 'later', 'm', 'km', 'face',
               'heres', "you'll", 'regards', "weren't", 'normally', 'fact', 'saying',
               'particularly', 'et-al', 'show', 'able', 'anyway', 'ending', 'find', 'promptly',
               'one', 'specifically', 'mug', "won't", 'should', 'only', 'going', 'specify',
               'announce', 'pointed', "there've", 'do', 'over', 'his', 'above', 'means',
               'between', 'stop', 'sensible', 'truly', "they'd", 'ones', 'hid', 'nearly',
               'words', 'despite', 'during', 'beings', 'him', 'is', 'areas', 'regarding',
               'qv', 'h', 'generally', 'twice', 'she', 'contain', 'x', 'where', 'rooms',
               'ignored', 'their', 'ends', "hasn't", 'namely', 'sec', 'are', "that've",
               'best', 'wonder', 'said', 'ways', 'away', 'currently', 'please', 'ups',
               "wasn't", 'outside', "there's", 'various', 'hopefully', 'affecting', 'probably',
               'neither', 'across', 'available', 'we', 'never', 'recently', 'opening',
               'importance', 'points', 'however', 'by', 'no', 'come', 'both', 'c', 'last',
               'thou', 'many', 'taking', 'thence', 'according', 'against', 'etc', 's',
               'became', 'interesting', 'com', 'asked', 'comes', 'otherwise', 'among',
               'presumably', 'co', 'ZZ', 'point', 'within', 'had', 'ca', 'whatever',
               'furthered', 'ZT', "couldn't", 'moreover', 'throughout', 'considering', 'meantime',
               'pp', 'described', 'asks', "it's", 'due', 'been', 'quickly', 'whom', 'much',
               'interest', 'certain', 'whod', 'hardly', "it'd", 'wants', 'adopted',
               'corresponding', 'beforehand', "what's", 'else', 'finds', 'worked', 'an', 'hers',
               'former', 'those', 'case', 'myself', 'novel', 'look', 'unlike', 'these',
               'thereto', 'value', 'n', 'will', 'while', "wouldn't", 'theres', 'seven',
               'whereafter', 'almost', 'wherever', 'refs', 'thus', 'it', 'cant', 'someone',
               'im', 'in', 'somebody', 'alone', 'id', 'if', 'different', 'anymore',
               'perhaps', 'suggest', 'make', 'same', 'wherein', 'member', 'parts',
               'potentially', 'widely', 'several', 'howbeit', 'used', 'see', 'somewhere',
               'keys', 'faces', 'upon', 'effect', 'uses', 'interested', 'thoughts', 'recent',
               'off', 'whereby', 'older', 'nevertheless', 'makes', 'youre', 'well', 'kept',
               'obviously', 'thought', 'without', "can't", 'y', 'the', 'yours', 'latest',
               'lest', 'things', "she'll", 'newest', 'just', 'less', 'being', 'nd',
               'therere', 'liked', 'beginnings', 'thanks', 'behind', 'facts', 'useful',
               'yes', 'lately', 'yet', 'unto', 'afterwards', 'wed', "we've", 'seems',
               'except', 'thousand', 'lets', 'other', 'inner', 'tell', 'has', 'adj', 'ought',
               'gave', "t's", 'around', 'big', 'showing', "who's", 'possible', 'usefully',
               'early', 'possibly', 'five', 'know', 'similarly', 'world', 'apart', 'name',
               'abst', 'nay', 'necessary', 'like', 'follows', 'theyre', 'either', 'fully',
               'become', 'works', 'page', 'grouping', 'therein', 'shed', 'because', 'old',
               'often', 'successfully', 'some', 'back', 'self', 'towards', 'shes', 'specified',
               'home', "'ve", 'thinks', 'happens', 'vol', "there'll", 'for', 'affects',
               'highest', 'though', 'per', 'whole', 'everything', 'asking', 'provides', 'tends',
               'three', 't', 'be', 'who', 'run', 'furthers', 'seconds', 'of', 'obtain',
               'nowhere', 'although', 'entirely', 'on', 'about', 'goods', 'ok', 'would',
               'anything', 'oh', 'theirs', 'v', 'o', 'whomever', 'whence', 'important',
               'plus', 'act', 'slightly', 'or', 'seeing', 'own', 'whats', 'formerly',
               'previously', "n't", 'into', 'youd', 'www', 'getting', 'backing', 'hes',
               'appropriate', 'very', 'primarily', 'theyd', 'couldnt', 'whos', 'your', 'her',
               'area', 'aren', 'downed', 'apparently', 'there', 'long', 'why', 'hed',
               'accordingly', "we're", 'way', 'resulted', 'was', 'opens', 'himself',
               'elsewhere', 'enough', 'becoming', 'but', 'somehow', 'hi', 'ended', 'newer',
               'line', 'trying', 'with', 'he', 'usefulness', "they're", 'made', 'places',
               'mrs', 'whether', 'wish', 'j', 'up', 'us', 'throug', 'placed', 'below',
               'un', 'whim', 'problem', 'z', 'similar', 'noone', "we'd", 'strongly', 'gone',
               'sometimes', 'ordered', 'ah', 'describe', 'am', 'general', 'meanwhile', 'as',
               'sometime', 'right', 'at', 'our', 'work', 'inc', 'again', 'uucp', "'ll",
               'nonetheless', 'greetings', 'na', 'whereas', 'tip', 'backs', 'ourselves', 'til',
               'grouped', 'definitely', 'latterly', 'wheres', 'you', 'really', 'concerning',
               'showns', 'briefly', "'t", "'s", 'regardless', 'welcome', 'problems', "let's",
               'sure', "'d", "'m", 'sides', 'began', 'younger', 'e', 'longer', 'using',
               'came', 'backed', 'together', 'hello', 'itself', 'u', 'presenting', 'serious',
               'evenly', 'orders', 'once'}
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
    #print("aaaa")
    review = decontracted(review.lower())

    processed_review = [word for word in review.lower().translate(
                                            str.maketrans('', '',
                                            string.punctuation)).split() if word not in stop_words]
    if len(processed_review) < MAX_WORDS_IN_REVIEW:
        replace = [' ']*(MAX_WORDS_IN_REVIEW - len(processed_review))
        processed_review.extend(replace)
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
    input_data = tf.placeholder(tf.float32, [None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
    #input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    # with tf.name_scope("labels"):
    #labels = tf.placeholder(tf.int32,[None, NUM_CLASSES],name='labels')
    labels = tf.placeholder(tf.int32, [None, NUM_CLASSES], name='labels')
    # with tf.name_scope("dropout_keep_prob"):
    dropout_keep_prob = tf.placeholder_with_default(0.5, shape=(),name='dropout_keep_prob')
    # dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
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
