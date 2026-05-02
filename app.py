from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

app = Flask(__name__)

tweets_data = []
x = []
y = []
vectorizer = CountVectorizer(stop_words='english')
model_trained = False
dtree = None

def load_data():
    global tweets_data, x, y, model_trained, dtree
    
    tweets_data = []
    x = []
    y = []
    
    try:
        with open('data/tweetdata.txt', 'r') as f:
            for line in f:
                try:
                    tweet = json.loads(line)
                    tweets_data.append(tweet)
                except:
                    continue

        sent = pd.read_excel('processed_data/output.xlsx')
        
        for i in range(len(tweets_data)):
            if tweets_data[i]['id'] == sent['id'][i]:
                x.append(tweets_data[i]['text'])
                y.append(sent['sentiment'][i])

        train_features = vectorizer.fit_transform(x)
        dtree = tree.DecisionTreeClassifier()
        dtree.fit(train_features, [int(r) for r in y])
        model_trained = True
        print("Model trained successfully!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        model_trained = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model_trained:
        return jsonify({'error': 'Model not ready yet!'})
    
    data = request.get_json()
    tweet = data.get('tweet', '')
    
    if not tweet.strip():
        return jsonify({'error': 'Please enter some text!'})
    
    try:
        input_features = vectorizer.transform([tweet])
        prediction = dtree.predict(input_features)[0]
        
        if prediction == 1:
            result = 'Positive'
            emoji = '😊'
            message = 'This tweet seems happy and healthy!'
            color = 'positive'
        elif prediction == 0:
            result = 'Neutral'
            emoji = '😐'
            message = 'This tweet seems neither happy nor sad.'
            color = 'neutral'
        else:
            result = 'Negative'
            emoji = '😔'
            message = 'This tweet may show signs of sadness or depression.'
            color = 'negative'
            
        return jsonify({
            'result': result,
            'emoji': emoji,
            'message': message,
            'color': color
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_data()
    app.run(debug=True)