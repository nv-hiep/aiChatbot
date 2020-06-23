import pickle
import json
import random

import numpy             as np

from keras.models        import load_model

# For tkinter GUI
import tkinter
from tkinter             import *

import nltk
from nltk.stem           import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
stemmer    = LancasterStemmer()

# Load model
model   = load_model('models/chatbot_model.h5')

# Load data from files
intents = json.loads(open('data/intents.json').read())
words   = pickle.load(open('data/words.pkl','rb'))
classes = pickle.load(open('data/classes.pkl','rb'))



# def clean_up_sentence(sentence):
#     # tokenize the pattern - splitting words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # stemming every word - reducing to base form
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words



def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words





# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # bag of words  vocabulary matrix
    bag = [0]*len(words)  
    
    for s in sentence_words:
        for (i,w )in enumerate(words):
            if (w == s):
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
            # -
        # End - for
    # End - for

    return(np.array(bag))





def predict_class(sentence):
    # filter below  threshold predictions
    p         = bow(sentence, words)
    res       = model.predict(np.array([p]))[0]
    er_thresh = 0.25
    results   = [[i,r] for i,r in enumerate(res) if r > er_thresh]
    
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list




def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result




def send():
    msg = entry_box.get('1.0','end-1c').strip()
    entry_box.delete('0.0',END)

    if msg != '':
        chat_box.config(state=NORMAL)
        chat_box.insert(END, 'You: ' + msg + '\n\n')
        chat_box.config(foreground='#446665', font=('Verdana', 12 ))
    
        ints = predict_class(msg)
        res = get_response(ints, intents)
        
        chat_box.insert(END, 'Bot: ' + res + ' (' + ints[0]['intent'] + ', '+ str(round(float(ints[0]['probability'])*100.,1)) +'%)' + '\n\n')
            
        chat_box.config(state=DISABLED)
        chat_box.yview(END)
 







#---------- MAIN ----------#

# Test --- 
# msg = 'hi'
# ints = predict_class(msg)
# res = get_response(ints, intents)
# print(ints)
# print( res )

root = Tk()
root.title('Chatbot')
root.geometry('500x600')
root.resizable(width=FALSE, height=FALSE)

#Create Chat window
chat_box = Text(root, bd=0, bg='white', height='8', width='50', font='Arial')
chat_box.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=chat_box.yview, cursor='heart')
chat_box['yscrollcommand'] = scrollbar.set

#Create Button to send message
send_button = Button(root, font=('Verdana',12,'bold'), text='Send', width='12', height=5,
                    bd=0, bg='#f9a602', activebackground='#3c9d9b',fg='#000000',
                    command= send )

#Create the box to enter message
entry_box = Text(root, bd=0, bg='white',width='29', height='5', font='Arial')
#entry_box.bind('<Return>', send)


#Place all components on the screen
scrollbar.place(x=476,y=6, height=486)
chat_box.place(x=6,y=6, height=386, width=470)
entry_box.place(x=128, y=501, height=50, width=370)
send_button.place(x=6, y=501, height=50)

root.mainloop()