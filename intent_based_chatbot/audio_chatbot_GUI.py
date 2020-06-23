#!/usr/bin/env python
__author__ = 'Hiep Nguyen'

import os
import pickle
import json
import random
import sys

try:               # Python 2.7x
    import Tkinter as tk
    import ttk
except Exception:  # Python 3.x
    import tkinter as tk
    from tkinter import ttk





import numpy             as np

from keras.models        import load_model

# For tkinter GUI
import tkinter
from tkinter             import *

import nltk
from nltk.stem           import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

# Import the required module for text  
# to speech conversion 
from gtts                import gTTS 
from time                import sleep


# Download packages/files
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
stemmer    = LancasterStemmer()

model   = load_model('chatbot_model.h5')

intents = json.loads(open('data/intents.json').read())
words   = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(tk.Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        tk.Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master       = master
        
        self.bg_colour    = '#ececec'
        self.array        = ''
        self.C            = 2.99792458e8 # m/s - speed of light
        self.scrollbar    = None
        self.chat_box     = None
        self.entry_box    = None
        self.send_button  = None
        self.close_button = None
        self.msg          = ''
        self.res          = ''

        # changing the title of our master widget      
        self.master.title('Chatbot')
        
        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()




    





    #Creation of init_window
    def init_window(self):

        # creating a menu instance
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)


        # create the file object)
        file = tk.Menu(menu, tearoff=True)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label='Exit', command=self.quit)

        #added "file" to our menu
        menu.add_cascade(label='File', menu=file)

        



        # create the file object)
        edit = tk.Menu(menu, tearoff=True)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label='Undo')

        #added "file" to our menu
        menu.add_cascade(label='Edit', menu=edit)




        #Create Chat window
        self.chat_box = Text(self.master, bd=0, bg='white', height='8', width='50', font='Arial')
        # # chat_box = tk.Text(self.master, bd=0, bg='white', height='8', width='50', font='Arial')
        self.chat_box.config(state=DISABLED)

        # Bind scrollbar to Chat window
        self.scrollbar = Scrollbar(self.master, command=self.chat_box.yview, cursor='heart')
        self.chat_box['yscrollcommand'] = self.scrollbar.set


        # Create Button to send message
        self.send_button = Button(self.master, font=('Verdana',12,'bold'), text='Send',
                             width=12, height=5, bd=0,
                             bg='#f9a602', activebackground='#3c9d9b', fg='#000000',
                             command=self.send )

        #Create the box to enter message
        self.entry_box = Text(self.master, bd=0, bg='white',width='29', height='5', font='Arial')


        # self.close_button = Button(self.master, text="Close", 
        #                         bg='#F72C2C', activebackground='#F72C2C', fg='#000000',
        #                         command=self.quit)


        #Place all components on the screen
        self.scrollbar.place(x=476,y=6, height=486)
        self.chat_box.place(x=6,y=6, height=386, width=470)
        self.entry_box.place(x=128, y=501, height=50, width=370)
        self.send_button.place(x=6, y=501, height=50)
        # self.close_button.place(x=6, y=555, height=50)
    # def init_window

    





    def quit(self):
        exit()





    def send(self):
        self.msg = self.entry_box.get('1.0','end-1c').strip()
        self.entry_box.delete('0.0', END)

        if self.msg == '':
            return

        self.chat_box.config(state=NORMAL)
        self.chat_box.insert(END, 'You: ' + self.msg + '\n\n')
        self.chat_box.config(foreground='#446665', font=('Verdana', 12 ))

        ints     = self.predict_class()
        self.res = self.get_response(ints, intents)
        
        self.chat_box.insert(END, 'Bot: ' + self.res + ' (' + ints[0]['intent'] + ', '+ str(round(float(ints[0]['probability'])*100.,1)) +'%)' + '\n\n')
        # Speak out the response
        # self.speak(res)

        self.chat_box.config(state=DISABLED)
        self.chat_box.yview(END)

        # Speak out the message
        if(self.res != ''):
            self.speak()



    def clean_up_sentence(self):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(self.msg)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words





    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, words):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence()
        
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





    def predict_class(self):
        # filter below  threshold predictions
        p         = self.bow(words)
        res       = model.predict(np.array([p]))[0]
        
        er_thresh = 0.25
        results   = [[i,r] for i,r in enumerate(res) if r > er_thresh]
        
        # sorting strength probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        
        return return_list




    def get_response(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']

        ret = ''
        for intent in list_of_intents:
            if(intent['tag']== tag):
                ret = random.choice(intent['responses'])
                break
        return ret
 





    def speak(self):          
        # Passing the text and language to the engine,  
        # here we have marked slow=False. Which tells  
        # the module that the converted audio should  
        # have a high speed 
        myobj = gTTS(text=self.res, lang='en', slow=False) 
          
        # Saving the converted audio in a mp3 file named 'temp'
        temp = 'temp.mp3'
        myobj.save(temp) 
          
        # Playing the converted file 
        os.system('mpg123 ' + temp)

        # Remove the .mp3 file
        os.remove(temp)
# End - class Window



#---------- MAIN ----------#

# root window created. Here, that would be the haveonly window, but
# you can later have windows within windows.
root = Tk()
root.title('Chatbot')
root.geometry('500x600')
root.resizable(width=FALSE, height=FALSE)
# root.minsize(640, 100)

# creation of an instance
Window(root)

# mainloop 
root.mainloop()