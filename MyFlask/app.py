from flask import Flask, render_template, request
import numpy as np

from pickle import load
from sklearn.preprocessing import Normalizer
#from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
from TraiterImage import extract_features
import warnings
warnings.filterwarnings("ignore")

clf = load(open('modele_clf.pkl', 'rb'))
features = load(open('deploiement_featuresA.pkl', 'rb'))


train = np.array(features)

scaler = Normalizer()
scaler.fit(train)

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('index.html')


@app.route('/', methods = ['POST'])
def upload_file():
	if request.method == 'POST':
		img = request.files['image']


		img.save("static/"+img.filename)

		caption = prediction("static/"+img.filename)
        



		
		result_dic = {
			'image' : 'static/' + img.filename,
			'description' : caption
		}
	return render_template('index.html', results = result_dic)

def prediction(imgt):
    
    
    ft_test = extract_features([imgt])
    caption_list = []
    caption_ret = []
    
    for img in ft_test:

        img =scaler.transform(img.reshape(1,-1))
        pred = clf.predict(img.reshape(1,-1))
        
        caption_tokens = [t.lower() for t in word_tokenize(pred[0])]
        caption = " ".join(caption_tokens)
        if len(caption) >0 :
            caption_list.append(caption)
            
    # remove predictions which are the same
    caption_list = set(caption_list)   
    captions = ""
    for capt in caption_list :
        caption_ret.append(capt)
    if len(caption_ret) <1 :
        captions = "We have detected nothing."+"<br>"+" Maybe the image is not in our domain, Sorry !!!"
        
    else :
        captions = captions + caption_ret[0]
        for i in range(len(caption_ret)-1) :
            captions = captions + " ,"+"<br>"+ caption_ret[i+1]
       
    return captions
            
if __name__ == "__main__":
    app.run(debug=True)
    















