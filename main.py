from tensorflow import keras
import random
import tensorflow as tf
import re
from string import digits
import json

#This function takes an entire text (lowercased and no punctuation) of a text and then separates it into words for processing
def clean_data(filename, split_style=" "):
    with open (filename, "r", encoding="utf-8") as f:
        text = f.read().split(" ")
    return (text)

#This function creates a bag of words based on the two texts, i.e. Scripture and non-Scripture
def create_bwords(text1, text2):
    total_words = text1+text2
    word_index = list(dict.fromkeys(total_words))
    bwords = {}
    x=4
    for word in word_index:
        bwords[word] = x
        x=x+1
    return (bwords)

#This function vectorizes the words from a text. I recommend combining both texts and then processing them simultaneously.
def vectorize(text, bwords):
    vector_text = []
    for word in text:
        num_word = bwords[word]
        vector_text.append(num_word)
    return (vector_text)

#This function separates a text into predetermined chunks
def create_chunks(text_array, chunk):
    chunks = [text_array[i:i + chunk] for i in range(0, len(text_array), chunk)]
    final = []
    for item in chunks:
        if len(item) == chunk:
            final.append(item)

    return (final)

#This function will allow you to create a reverse word index. I structured this off Tech with Tim's neural network lessons.
def reverse_index(bwords):
    reverse_word_index = {value : key for (key, value) in bwords.items()}
    return (reverse_word_index)

#This function reconstructions a text based on the reverse_index function.
def reconst_text(text, reverse_word_index):
    return (" ".join([reverse_word_index.get(i, "?") for i in text]))

#This function prepares the data with the correct labels. Feed one text at a time and assign it an integer of 0 or 1.
def prepare_data(chunks, label):
    total_chunks = []
    for chunk in chunks:
        total_chunks.append((chunk, label))
    return (total_chunks)

#This function processes the chunks into training data for the neural network.
def create_training(total_chunks, cutoff):
    #randomize our data
    random.shuffle(total_chunks)
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    test_num = len(total_chunks)*cutoff
    x=0
    for entry in total_chunks:
        if x > test_num:
            testing_data.append(entry[0])
            testing_labels.append(entry[1])
        else:
            training_data.append(entry[0])
            training_labels.append(entry[1])
        x=x+1
    return (training_data, training_labels, testing_data, testing_labels)

#This is our neural network model. I have left a few examples of other optional layers.
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(120000,16))
    model.add(keras.layers.GlobalAveragePooling1D())
    # model.add(keras.layers.Dropout(.1))
    # model.add(keras.layers.Dense(16,activation="relu"))
    model.add(keras.layers.Dense(16,activation="tanh"))
    # model.add(keras.layers.Dense(16,activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return (model)

#This function trains our model.
def train_model(model, tt_data, val_size=.3, epochs=1, batch_size=16, save_model=False, save_r=True, save_bwords=False,chunk=5):
    vals = int(len(tt_data[0])*val_size)
    training_data = tt_data[0]
    training_labels = tt_data[1]
    testing_data = tt_data[2]
    testing_labels = tt_data[3]
    x_val = training_data[:vals]
    x_train = training_data[vals:]

    y_val = training_labels[:vals]
    y_train = training_labels[vals:]

    fitModel = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=1)
    model_results = model.evaluate(testing_data, testing_labels)
    print (model_results)
    main_model_data = f"FUNCTIONTEST_vulgate_model_{str(chunk)}_{str(epochs)}_{str(batch_size)}_v1.3.h5"
    if save_model==True:
        model_name = f"models/tests/FUNCTIONTEST{main_model_data}"
        model.save(model_name)
    if save_r==True:
        results_file = f"models/tests/FUNCTIONTEST{main_model_data}_results"
        save_results(results_file, model, epochs=epochs, batch_size=batch_size, chunk=chunk, model_results=model_results)
    if save_bwords == True:
        json_bwords_name = f"models/tests/FUNCTIONTEST{main_model_data}_bwords.json"
        with open (json_bwords_name, "w") as json_file:
            json.dump(bwords, json_file, indent=4)
    return (model, results_file)

#This function saves the results in a text file.
def save_results(filename, model, batch_size, epochs, chunk, model_results, cutoff=0):
    with open (f"{str(filename)}.txt", "w", encoding="utf-8") as f:
        f.write("========================MODEL SUMMARY============================")
        f.write("\n")
        f.write("Saved Model Name: " + str(cutoff))
        f.write("\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")
        f.write("\n")
        f.write("=================================================================")
        f.write("\n")
        f.write("Number of Epochs: " + str(epochs))
        f.write("\n")
        f.write("Batch Size: " + str(batch_size))
        f.write("\n")
        f.write("Text Chunk Size: " + str(chunk))
        f.write("\n")
        f.write("Results Cutoff: " + str(cutoff))
        f.write("\n")
        f.write("========================MODEL ACCURACY===========================")
        f.write("\n")
        f.write("\n")
        f.write(str(model_results))
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("========================END OF SUMMARY===========================")
        f.write("\n")
        f.write("\n")
        f.write("========================MODEL RESULTS============================")
        f.write("\n")
        f.write("\n")


#This function prepares a text to test the model on
def prepare_source(filename, vects):
    with open (filename, "r", encoding="utf-8") as f:
        text = f.read()
    bwords = []
    sentences = re.split(r'[.:;?]\s*', text)
    for sentence in sentences:
        words = sentence.split(" ")
        new_words = []
        for word in words:
            word = re.sub(r"[\(\[].*?[\)\]]", "", word).replace(" ,", ",").replace(" .", ".").replace(" .", ".").replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n").replace("  ", " ").replace("  ", " ").replace("\n ", "\n").replace(" ;", ";").replace(" :", ":").replace(" ?", "?").replace("?  ", "? ").replace(" ,", ",").replace("J", "I").replace("j", "i").replace(",", "").replace("\n", "").replace(":", "").replace(";", "").lower()
            remove_digits = str.maketrans('', '', digits)
            word = word.translate(remove_digits)
            if word in vects.keys():
                num_word=vects[word]
            else:
                num_word = 119000
            new_words.append(num_word)
        bwords.append(new_words)
    return (bwords)

#This function runs the test on the test text
def test_model(text_chunks, reverse_word_index, model, cutoff=0):
    results = []
    print ("Scripture Found: ")
    for test in text_chunks[47:100]:
        if len(test)>2:
            print (test)
            predict = model.predict([test])
            if predict[0] > cutoff:
                # print (decode_review(test_review))
                print ("Prediction: " + str(predict[0]))
                results.append((str(predict[0]), reconst_text(test, reverse_word_index)))
    return (results, cutoff)

#This function outputs the results into the same text file as above.
def write_test(results, filename):
    with open (filename+".txt", "a", encoding="utf-8") as f:
        for result in results:
            f.write(str(result)+"\n")






###Sample of Using these Functions for scripture
chunk = 5
epochs = 1
batch_size=16

#files for the 3 sources: scripture, non-scripture, and a test source
scripture_file = "data/scripture/vulgate_cleaned_lp.txt"
nonscripture_file = "data/caesar/caesar_gal_cleaned_lp.txt"
raban_file_clean = "data/rabanus/rabanus_com_matt_cleaned.txt"

###cleaning data for training
clean_scripture = clean_data(filename=scripture_file, split_style="words")
clean_nonscripture = clean_data(filename=nonscripture_file, split_style="words")[0:len(clean_scripture)]

#convert data to bag of words
bwords = create_bwords(text1=clean_scripture, text2=clean_nonscripture)

#vectorize texts
vect_scripture = vectorize(clean_scripture, bwords)
vect_nonscripture = vectorize(clean_nonscripture, bwords)

#separate these texts into a desired input
chunk_scripture = create_chunks(text_array=vect_scripture, chunk=chunk)
chunk_nonscripture = create_chunks(text_array=vect_nonscripture, chunk=chunk)

#label data
label_scripture = prepare_data(chunks=chunk_scripture, label=1)
label_nonscripture = prepare_data(chunks=chunk_nonscripture, label=0)

#combine data
total_chunks = label_scripture+label_nonscripture

#create training data
tt_data = create_training(total_chunks, cutoff=.95)

#create model
model = create_model()

#train model
model = train_model(model, tt_data=tt_data, chunk=chunk, epochs=epochs, batch_size=batch_size)

#reverse word_index
reverse_word_index = reverse_index(bwords)

#Running the test on Rabanus
prep_raban = prepare_source(raban_file_clean, vects=bwords)
test_results = test_model(prep_raban, reverse_word_index=reverse_word_index, model=model[0])
write_test(test_results[0], filename=str(model[1]))

