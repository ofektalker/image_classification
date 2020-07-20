
import numpy as np
import io
import pickle
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Activation
import argparse
from PIL import Image
import imageio
from sklearn import metrics
############################################
parser = argparse.ArgumentParser()
parser.add_argument('--task',
                    metavar='Query',
                    choices=['train', 'test'],
                    type=str, help='Choose between train / test')
parser.add_argument('--image',
                    metavar='Image to classify',
                    type=str,
                    help='File containing image for classification. Used with --test')
parser.add_argument('--model',
                    metavar='Model',
                    type=str,
                    help='File containing Trained model')
args = parser.parse_args()
############################################
CIFAR100_META_PATH = 'cifar-100-python/meta'
CIFAR100_TRAIN_PATH = 'cifar-100-python/train'
CIFAR100_TEST_PATH = 'cifar-100-python/test'
TARGET_MODEL_NAME = 'image_model.h5'
############################################
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data
############################################
word_vectors = load_vectors('wiki-news-300d-1M.vec')
############################################
def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
############################################
def word2vec_labels():
    # expects labels to be a list
    # each member of labels is a list of words (can be lenght 1)
    # process each member into vector of 300 floats
    
    meta_data = unpickle(CIFAR100_META_PATH)
    
    labels = np.array([name.decode('utf8') for name in meta_data[b'fine_label_names']]) # named labels
    split_labels = [label.split('_') for label in labels] # some labels have multiple words
    # process labels into Word Vectors, use MEAN on multi word labels
    
    vectorized_labels = []
    for label in split_labels:
        if len(label) > 1:
            vectorized_labels.append(np.mean([word_vectors[word] for word in label], axis=0))
        else:
             vectorized_labels.append(word_vectors[label[0]])
    return vectorized_labels
############################################
# # TRAIN SECTION
def prep_cifar_100_data():

    train_dict = unpickle(CIFAR100_TRAIN_PATH)
    test_dict = unpickle(CIFAR100_TEST_PATH)
    
    vectorized_labels = word2vec_labels()
    
    data_train = np.array(train_dict[b'data']).astype('float32')
    # 50k records
    # each record: 3072 in length
    # first 1024 - red channel
    # next 1024 - green channel
    # last 1024 - blue channel
    # reshape images data to represent the actual images of size 32x32 in rgb format (3 channels)
    data_train =  np.transpose(np.reshape(data_train, (-1, 32, 32, 3), order='F'), axes=(0, 2, 1, 3))
    # change label indexes into proper vectors
    labels_train = [vectorized_labels[t] for t in train_dict[b'fine_labels']]

    # repeat process from above for test data
    data_test = np.array(test_dict[b'data']).astype('float32')
    data_test = np.transpose(np.reshape(data_test, (-1, 32, 32, 3), order='F'), axes=(0, 2, 1, 3))
    labels_test = [vectorized_labels[t] for t in test_dict[b'fine_labels']]
    
    return [np.array(data_train), np.array(labels_train), np.array(data_test), np.array(labels_test)]
############################################
def construct_cnn(data_shape):
    no_filter = 32 # number of filters to use in convolutional layers
    
    no_conv_hidden_layers = 2 # how many convolutional layers to add on top of first layer
    no_regres_hidden_layers = 4 # how many regression hidden layers
    output_dim = 300 # output should be a vector of 300 floats
    conv_padding = 'same'
    cnn_activations = 'relu'
    conv_data_format = 'channels_last'
    feature_window_size = (3, 3)
    window_stride = 2
    pool = (2, 2)
    chanDim = -1
    conv_drop_rate = .25
    regres_drop_rate = .5
    optimizer_choice = 'adam'
    loss_func = 'mean_squared_error' # worked best, didn't go with categorical crossentropy because output dim != # of classes
    metrics_choice = ['accuracy']
    
    classifier = Sequential()
    # Step 1 - Convolution Layers -> Step 2 - Batch Normalizing -> Step 3 - Max Pooling -> Step 4 - Dropout
    classifier.add(Convolution2D(no_filter, 
                             kernel_size=feature_window_size, 
                             padding=conv_padding, 
                             input_shape=data_shape, 
                             activation=cnn_activations, 
                             data_format=conv_data_format)
              )
    classifier.add(BatchNormalization(axis=chanDim))
    classifier.add(MaxPooling2D(pool_size=pool, strides=window_stride))
    classifier.add(Dropout(conv_drop_rate))
    # repeat for as many convolution hidden layers as you want
    for _ in range(no_conv_hidden_layers):
        classifier.add(Convolution2D(no_filter, 
                             kernel_size=feature_window_size, 
                             padding=conv_padding,
                             activation=cnn_activations)
              )
        classifier.add(BatchNormalization(axis=chanDim))
        classifier.add(MaxPooling2D(pool_size=pool, strides=window_stride))
        classifier.add(Dropout(conv_drop_rate))
    # add flatten layer
    classifier.add(Flatten())

    # Regression layers
    offset = int(output_dim / no_regres_hidden_layers)
    for i in range(no_regres_hidden_layers):
        classifier.add(Dense(output_dim - offset * (no_regres_hidden_layers - (i + 1))))
        classifier.add(Activation(cnn_activations))
        classifier.add(BatchNormalization(axis=chanDim))
        classifier.add(Dropout(regres_drop_rate))
    
    classifier.compile(optimizer = optimizer_choice, loss = loss_func, metrics=metrics_choice)
    
    return classifier
############################################
def train():
    data_train, labels_train, data_test, labels_test = prep_cifar_100_data()
    classifier = construct_cnn(data_train[0].shape)
    classifier.fit(data_train, labels_train, epochs=25, batch_size=500)
    score, accuracy = classifier.evaluate(data_test, labels_test, batch_size=500)
    print("Model Score = {:.2f}".format(score))
    print("Accuracy = {:.2f}".format(accuracy*100))

    classifier.save(TARGET_MODEL_NAME)
############################################
# # TEST SECTION
def prep_image(image_path):
    from PIL import Image
    import imageio
    import numpy as np

    image_path_format = image_path.split('.') # search for format
    image_format = image_path_format[-1] # last dot marks the format
    image_name = '.'.join(image_path_format[:-1]) # all the rest is original filename
    image_correct_format_path = '%s_correct_format.%s'%(image_name, image_format) # path for corrected image
    
    img_info = Image.open(image_path, 'r')
    # convert to RGB if needed (some pics are RGBA, some PPA, whatever it is, we want RGB)
    if img_info.mode != 'RGB':
        img_info = img_info.convert('RGB')
    img_info.save(image_correct_format_path)
    # our system was trained on this size, so......
    if img_info.size != (32, 32):
        img_info = img_info.resize((32, 32))
    img_info.save(image_correct_format_path)
    # just so it's ready to be fed to CNN, just correct dimensions as expected, 1 image of 32x32, of 3 channels
    img_info = np.array(imageio.imread(image_correct_format_path))
    img_info = np.expand_dims(img_info, 0)
    
    return img_info
############################################
def get_prediction(image_path, model_path):
    
    from keras.models import load_model
    
    classifier = load_model(model_path)
    img_to_classify = prep_image(image_path)
    
    return classifier.predict(img_to_classify)
############################################

def cos_sim(a, b):

    from numpy import dot
    from numpy.linalg import norm

    return dot(a, b)/(norm(a)*norm(b))
############################################
def get_top_3_similar(prediction):
    
    vectorized_labels = word2vec_labels() # bring in all possible labels as vectors
    scores = metrics.pairwise.cosine_similarity(prediction, vectorized_labels)[0]
    # gather indexes of highest scores => corresponds to index of vectorized label
    top_3_similar_indexes = np.argsort(scores)[::-1][:len(scores)][:3]
    # get list of label where each label index corresponds to vectorized label index
    meta_data = unpickle(CIFAR100_META_PATH)
    labels = np.array([name.decode('utf8') for name in meta_data[b'fine_label_names']]) # named labels
    most_relevant = [labels[i] for i in top_3_similar_indexes]
    
    return most_relevant
############################################
def test(image_path, model_path):
    prediction = get_prediction(image_path, model_path)
    top_3_similar = get_top_3_similar(prediction)
    print('<%s> <%s> <%s>'%(top_3_similar[0], top_3_similar[1], top_3_similar[2]))
############################################
if args.task is not None:
    if args.task == 'train':
        train()
    elif args.task == 'test':
        if args.image is not None:
            if args.model is not None:
                test(args.image, args.model)
            else:
                print('please supply a model using \'--model <model_filename>.h5\'')
        else:
            print('please supply an image file using \'--image <image_filname>\'')
    else:
        print(
            'please choose between [\'train\', \'test\'] as parameter for --task')
else:
    print('please use \'--task <task>\'\n<task> is any of the values: (train, test)')