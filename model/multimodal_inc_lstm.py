"""
@belovm96:
    
Script for deep learning model - multimodal encoder-decoder architecture - Multimodal Inception LSTM Inject

"""
import tensorflow as tf
import os
import numpy as np

from keras import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from keras.activations import softmax
from keras.utils import to_categorical
from keras.optimizers import Adam

#Get the encoded captions
def get_captions(filename):
    img_cap = {}
    with open(filename,'r') as file:
        for line in file:
            line = line.strip()
            line_ = line.split(',')
            img_cap[line_[0]] = line_[1:]
            
    return img_cap

caption_path = '/workspace/google-cc-100k-with-threshold/word_to_vector_encoding.txt'
img_caption_dict = get_captions(caption_path)

#Load the encodings
enc_train = np.load("/home/jupyter/image-captioning-approaches/analysis/inception_enc_obj_train.npy")
enc_val = np.load("/home/jupyter/image-captioning-approaches/analysis/inception_enc_obj_validation.npy")

#Get the names of the images that we have the encoding for
with open('objects_in_images.txt', 'r') as infile:
    img_names = []
    for line in infile:
        line = line.strip()
        line = line.split('\t')
        img_names.append(line[0])
        
#Maximum sentence length     
MAX_LEN = 40
EMBEDDING_DIM = 300
IMAGE_ENC_DIM = 300

#Look up vocab size
vocab_size = 9555

#Encoded objects as input
img_input = Input(shape=(512,))
img_enc = Dense(300, activation="relu") (img_input)
images = RepeatVector(MAX_LEN)(img_enc)

#Embedding of partial input sequance
text_input = Input(shape=(MAX_LEN,))
embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)

#Concatenate encoded objects with partial input sequence
x = Concatenate()([images,embedding])
y = Bidirectional(LSTM(256, return_sequences=False))(x) 
pred = Dense(vocab_size, activation='softmax')(y)
model = Model(inputs=[img_input,text_input],outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

model.summary()


#Define training generator
def training_generator(batch_size=125):
    
    input_mat_seq = np.zeros((batch_size,MAX_LEN))
    input_mat_enc = np.zeros((batch_size,512))
    output_mat = np.zeros((batch_size,vocab_size))
    c = 0
    
    while True:
        enc_count = 0
        for image_name in img_names[:70000]:
            img_enc = enc_train[enc_count]
            enc_count += 1
            if image_name in img_caption_dict:
                caption = img_caption_dict[image_name]
                if len(caption) <= MAX_LEN:
                    input_part_seq = [0] * MAX_LEN
                    for i in range(1,len(caption)):
                        word_id_in = int(caption[i-1])
                        word_id_out = int(caption[i])
                        input_part_seq[i-1] = word_id_in
                        output_mat[c, word_id_out] = 1
                        input_mat_seq[c] = input_part_seq
                        input_mat_enc[c] = img_enc
                        c += 1
                        if c == batch_size:
                            yield ([input_mat_enc,input_mat_seq],output_mat)
                            c = 0
                            input_mat_seq = np.zeros((batch_size,MAX_LEN))
                            output_mat = np.zeros((batch_size,vocab_size))
                            input_mat_enc = np.zeros((batch_size,512))


batch_size = 125
train_gen = training_generator(batch_size)
steps_train = enc_train.shape[0] * MAX_LEN // batch_size

model.fit_generator(generator=train_gen, steps_per_epoch=steps_train, epochs=20, verbose=True)

model.save_weights("/home/jupyter/image-captioning-approaches/analysis/weights_obj_det_inc_lstm_inject.h5")