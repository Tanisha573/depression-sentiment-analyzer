from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

app = Flask(__name__)

# Built-in training data - no files needed!
training_tweets = [
    "I feel so sad and hopeless today nothing makes me happy anymore",
    "Feeling great today Life is beautiful and wonderful",
    "I have been feeling anxious and stressed about everything lately",
    "Today was an amazing day full of joy and happiness",
    "I cannot stop crying depression is taking over my life",
    "Feeling neutral about everything just another ordinary day",
    "I am so happy and grateful for everything in my life",
    "Anxiety and depression are ruining my life I feel so alone",
    "What a wonderful morning feeling blessed and positive",
    "I feel empty inside nothing brings me joy anymore",
    "Life is good today spending time with family and friends",
    "Feeling very depressed and lonely nobody understands me",
    "Just had the best day ever everything is going great",
    "I am struggling with dark thoughts and sadness every day",
    "Feeling okay today nothing special just a regular day",
    "So grateful and happy life could not be better right now",
    "Depression hits hard today feeling worthless and hopeless",
    "Had a productive day feeling accomplished and satisfied",
    "Feeling so low and miserable stress and anxiety everywhere",
    "Today is a beautiful day feeling happy and energetic",
    "I hate everything nothing ever goes right for me",
    "Life is absolutely wonderful I love every moment",
    "I am exhausted and burned out from all this stress",
    "Feeling motivated and ready to take on the world today",
    "Everything feels dark and pointless I cannot go on",
    "Had an incredible time with friends today so joyful",
    "I feel numb and disconnected from everything around me",
    "Woke up feeling refreshed and excited about today",
    "Nobody cares about me I feel invisible and worthless",
    "Feeling content and peaceful life is treating me well",
]

training_labels = [
    -1, 1, -1, 1, -1, 0, 1, -1, 1, -1,
    1, -1, 1, -1, 0, 1, -1, 1, -1, 1,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 0,
]

vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(training_tweets)
dtree = tree.DecisionTreeClassifier()
dtree.fit(train_features, training_labels)
print("Model trained successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
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
    app.run(debug=True)