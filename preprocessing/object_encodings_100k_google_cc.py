"""
@belovm96:
    
Script for object encoding in 100k subset of Google's Conceptual Captions dataset

Object encoder is Google's Universal Sentence Encoder, and the object prediction model is Inception (v.3)
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.applications.imagenet_utils import decode_predictions
import tensorflow_hub as hub
import os
import numpy as np
import PIL
from keras.applications.inception_v3 import InceptionV3
from keras import Model

#Using Google Universal Sentence Encoder
def embed_useT(module):
    
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
        
    return lambda x: session.run(embeddings, {sentences: x})

#Encode images and save the object predictions to a file
def caption_enc(img_list, obj_model, IMG_PATH):
    file = open('/home/jupyter/image-captioning-approaches/analysis/objects_in_images.txt','w') 
    lst_enc = []
    for i, img in enumerate(img_list):
        if i % 3000 == 0:
            print('Image', i)
        try:
            image = PIL.Image.open(os.path.join(IMG_PATH, img))
            image = np.asarray(image.resize((299,299))) / 255.0
            predict_obj = obj_model.predict(np.array([image]))
            top_labels = decode_predictions(predict_obj, top=5)
            obj_list = [label[1] for label in top_labels[0]]
            str_labels = " ".join(obj_list)
            lst_enc.append(str_labels.replace('_', ' '))
            file.write(img+'\t'+str_labels.replace('_', ' ')+'\n')
        except:
            print(img, 'not an image...')
    file.close()     
    return lst_enc

img_path = '/dataset/google-cc/100k/images'
list_100k_imgs = os.listdir(img_path)

#Load Inception (v.3) CNN and Google's Universal Sentence Encoder
img_model = InceptionV3(weights='imagenet')
new_input = img_model.input
new_output = img_model.output
obj_pred = Model(new_input, new_output)
embed = embed_useT('module_useT')

#Create the list of the objects detected in images
embed_imgs_100 = caption_enc(list_100k_imgs, obj_pred, img_path)

#Encode the objects detected in images
train_1 = embed_imgs_100[:45000]
train_2 = embed_imgs_100[45000:70000]
train_first = embed(train_1)
train_sec = embed(train_2)
enc_train = np.concatenate((train_first, train_sec))

val = embed_imgs_100[70000:79000]
enc_val = embed(val)

test = embed_imgs_100[79000:89000]
enc_test = embed(test)

#Save the encodings
np.save("/home/jupyter/image-captioning-approaches/analysis/inception_enc_obj_train.npy", enc_train)
np.save("/home/jupyter/image-captioning-approaches/analysis/inception_enc_obj_test.npy", enc_test)
np.save("/home/jupyter/image-captioning-approaches/analysis/inception_enc_obj_validation.npy", enc_val)