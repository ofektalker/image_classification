# image_classification

Convolutional Neural Network trained to classify images

## Input format

image_classification.py --task <train/test> --image <image file name> --model <model file name>

### For training

similarity.py --task train --data <training dataset in .csv format> --model <model target file>

creates image_model.h5 used later for classifying

### For testing

similarity.py --query <query file global path> --text <text file global path> --task test --model <trained model>

prints 3 text labels (words) most relevant to this image (from the most relevant to the least relevant):
<label 1> <label 2> <label 3>

- used https://fasttext.cc/docs/en/english-vectors.html -> wiki-news-300d-1M.vec.zip - pre trained word vectors trained using fast text on the english language.
- used https://www.cs.toronto.edu/~kriz/cifar.html -> CIFAR-100 python version - Labeled subsets of the 80 million tiny images dataset.
