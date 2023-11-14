import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
from keras.models import load_model
from tensorflow.python.util import deprecation  # To remove tensorflow deprecation warnings
import warnings

deprecation._PRINT_DEPRECATION_WARNINGS = False
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
lemmatizer = WordNetLemmatizer()

class modelTrain:
    def __init__(self):
        # Initialization of variables
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!']

    def loadIntents(self, intents_path=''):
        # Reads and returns intents from a JSON file
        data_file = open(intents_path).read()
        intents = json.loads(data_file)
        return intents

    def preprocess_save_Data(self, intents):
        # Preprocesses and saves data for training
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Add documents in the corpus
                self.documents.append((w, intent['tag']))

                # Add to the classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize and lower each word and remove duplicates
        self.words = [lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))

        # Sort classes
        self.classes = sorted(list(set(self.classes)))

        # Save data
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

        return self.words, self.classes

    def prepareTrainingData(self, words, classes):
        # Create training data
        training = []

        # Empty output array
        output_empty = [0] * len(classes)

        # Training set, bag of words for each sentence
        for doc in self.documents:
            # Initialize our bag of words
            bag = []

            # List of tokenized words for the pattern
            pattern_words = doc[0]

            # Lemmatize each word and create base word to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # Create our bag of words array with 1 if word match found in current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # Output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # Find the maximum length of bag and output_row
        max_bag_length = max(len(item[0]) for item in training)
        max_output_row_length = max(len(item[1]) for item in training)

        # Ensure all bags and output_rows have the same length
        for item in training:
            item[0] += [0] * (max_bag_length - len(item[0]))
            item[1] += [0] * (max_output_row_length - len(item[1]))

        # Shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training, dtype=object)  # Specify dtype=object to handle varying lengths

        # Create train and test lists. X - patterns, Y - intents
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        return train_x, train_y

    def createModel(self, train_x, train_y, epochs=200, batch_size=5, save_path='model.model'):

        """
        Creates a model.
        Parameters:
            train_x --> x train values
            train_y --> y train values
            epochs --> no of epochs to be trained
            batch_size --> batch_size during training
            save_path --> path to save the created model

        Model Structure:
        Layer 1 - 128 neurons,      'relu' activation
        Layer 2 - 64 neurons,       'relu' activation
        Layer 3 - (no. of classes), 'softmax' activation
        Optimizer - Stochastic Gradient Descent --> (best for this example)
          learning rate: 0.01
          momentum: 0.9
          nesterov accelerated --> True
        Loss: categorical crossentropy
        """

        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
        model.save(save_path, hist)
        print("Model Successfully Created and saved")
        return model


class modelPredict:
    def __init__(self, intents_path='filename.json', model_path='model_name.json'):
        self.intents_path = intents_path
        self.model = model_path

    def clean_up_sentence(self, sentence):
        lemmatizer = WordNetLemmatizer()

        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)

        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("Found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence, model, error_threshold=0.25):
        ERROR_THRESHOLD = error_threshold
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        # filter out predictions below a threshold
        p = self.bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        # ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints, intents_json):
        import random
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, msg):
        """
        Outputs a response from the model.
        Pass in the input text to receive a response from the model.
        Parameters:
        msg --> The required input text
        """
        model = load_model(self.model)
        intents = json.loads(open(self.intents_path).read())
        ints = self.predict_class(msg, model)
        res = self.getResponse(ints, intents)
        return res
    # response_from_bot = chatbot_response(input_query)

    # Initialize trainer
trainer = modelTrain()
intents = trainer.loadIntents('intents.json')  # The path where the intents.json file is saved
words, classes = trainer.preprocess_save_Data(intents)  # Prepares and saves preprocessed word data
train_x, train_y = trainer.prepareTrainingData(words, classes)  # Prepares training data

# Create the model
model = trainer.createModel(train_x, train_y, save_path='cbv_model.model')

# Initialize predictor
predictor = modelPredict('intents.json', 'cbv_model.model')

# A test loop
running = True
while running:
    msg = input('You: ')
    if msg == 'quit':
        running = False
    else:
        response = predictor.chatbot_response(msg)
        print('Bot: ', response)
