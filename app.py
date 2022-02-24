#!/usr/bin/env python
# coding: utf-8

# In[12]:


from flask import Flask, render_template,request
import pickle
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# In[13]:


app = Flask(__name__)
model_1 = pickle.load(open('model.pkl','rb'))
count_tf_idf = pickle.load(open('model_tf_idf.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Predict', methods=['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    pattern = r"[^a-zA-Z']"
    int_features = re.sub(pattern, " ", str(int_features))
    int_features = int_features.lower()
    lemmas_final=[]    
    tokens = word_tokenize(int_features)
    lemmatizer  = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token, pos='v') for token in tokens] 
    lemmas_final.append(' '.join(lemmas))

    final = count_tf_idf.transform(lemmas_final)

    prediction_proba= model_1.predict_proba(final)[:, 1]
    if prediction_proba > 0.5:
        output = 'positive'
        probability = round(float(prediction_proba)*100,2)
    else:
        output = 'negative'
        probability = 100-round(float(prediction_proba)*100,2)

    
    return render_template('index.html', prediction_text=r"The text: '{}' is {}. The probability that the text is {} is {}%".format(int_features,output,output,probability))



# In[14]:


if __name__ == "__main_":
    app.run(debug=True)

