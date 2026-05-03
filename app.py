from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'mindx_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mindx.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =====================
# DATABASE MODELS
# =====================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    posts = db.relationship('Post', backref='author', lazy=True)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(280), nullable=False)
    sentiment = db.Column(db.String(20), default='neutral')
    sentiment_emoji = db.Column(db.String(10), default='😐')
    sentiment_color = db.Column(db.String(20), default='neutral')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    likes = db.relationship('Like', backref='post', lazy=True, cascade='all, delete-orphan')
    comments = db.relationship('Comment', backref='post', lazy=True, cascade='all, delete-orphan')

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(280), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    user = db.relationship('User', backref='comments')

# =====================
# AI MODEL
# =====================

training_tweets = [
    # POSITIVE
    "i feel happy today",
    "i am so happy today",
    "feeling happy and good",
    "i feel great today",
    "i am feeling wonderful",
    "today is a great day",
    "i am so excited",
    "life is beautiful",
    "i feel blessed today",
    "feeling amazing and grateful",
    "i love my life",
    "so much joy today",
    "feeling fantastic today",
    "i am grateful for everything",
    "today is wonderful",
    "i feel energetic and motivated",
    "had an amazing day",
    "feeling positive and hopeful",
    "i am happy and content",
    "life is good today",
    "spending time with loved ones",
    "feeling refreshed and excited",
    "i woke up feeling great",
    "everything is going well",
    "i feel confident today",
    "today was productive and fun",
    "i feel loved and appreciated",
    "smiling and feeling good",
    "feeling cheerful today",
    "i had a wonderful time",
    "i am joyful and at peace",
    "feeling on top of the world",
    "i feel strong and capable",
    "today made me smile so much",
    "i feel warm and loved",

    # NEGATIVE
    "i feel sad today",
    "i am feeling sad",
    "i feel so sad and hopeless",
    "i cannot stop crying",
    "depression is taking over my life",
    "i feel so alone",
    "i feel empty inside",
    "nothing brings me joy anymore",
    "feeling very depressed and lonely",
    "nobody understands me",
    "i am struggling with dark thoughts",
    "feeling worthless and hopeless",
    "i hate everything",
    "nothing ever goes right for me",
    "i am exhausted and burned out",
    "everything feels dark and pointless",
    "i feel numb and disconnected",
    "nobody cares about me",
    "i feel invisible and worthless",
    "i feel so alone and miserable",
    "i am so depressed and anxious",
    "everything hurts and nothing helps",
    "i want to disappear",
    "feeling hopeless and worthless",
    "i am broken and lost",
    "life feels pointless and empty",
    "i cannot sleep i feel terrible",
    "i am overwhelmed and falling apart",
    "feeling like nobody loves me",
    "i am scared and helpless",
    "i cry every day",
    "i feel useless and unwanted",
    "i have no energy",
    "i feel dead inside",
    "i am suffering every day",
    "i feel like giving up",
    "feeling hopeless about my future",
    "i am in pain and feel terrible",
    "i feel miserable every day",
    "life is not worth living",

    # NEUTRAL
    "today was just a normal day",
    "nothing special happened today",
    "i went to work and came back",
    "just another regular day",
    "feeling okay i guess",
    "today was average nothing exciting",
    "i did my routine things today",
    "nothing much going on today",
    "it was an okay kind of day",
    "i feel neither good nor bad",
    "today was pretty normal",
    "just chilling and doing nothing",
    "feeling indifferent about today",
    "today was fine nothing more",
    "i had a regular day at work",
    "same as yesterday nothing new",
    "feeling neutral about everything",
    "just an ordinary Wednesday",
    "today passed by quickly",
    "nothing bad nothing great today",
]

training_labels = [
    # POSITIVE - 35
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    # NEGATIVE - 40
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    # NEUTRAL - 20
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

# Using keyword-based approach for better accuracy
POSITIVE_WORDS = [
    'happy', 'great', 'wonderful', 'amazing', 'fantastic', 'joy', 'joyful',
    'blessed', 'grateful', 'love', 'excited', 'cheerful', 'good', 'positive',
    'beautiful', 'excellent', 'energetic', 'motivated', 'confident', 'content',
    'peaceful', 'smile', 'smiling', 'laugh', 'fun', 'incredible', 'awesome',
    'delighted', 'proud', 'hope', 'hopeful', 'bright', 'warm', 'enjoy'
]

NEGATIVE_WORDS = [
    'sad', 'hopeless', 'depressed', 'depression', 'anxiety', 'anxious',
    'alone', 'lonely', 'empty', 'worthless', 'useless', 'miserable',
    'terrible', 'awful', 'crying', 'cry', 'pain', 'hurt', 'suffering',
    'suffer', 'dark', 'pointless', 'numb', 'broken', 'lost', 'scared',
    'helpless', 'tired', 'exhausted', 'overwhelmed', 'stress', 'stressed',
    'hate', 'angry', 'fear', 'failure', 'failed', 'disappear', 'dead',
    'die', 'death', 'kill', 'suicide', 'give up', 'falling apart'
]

def analyze_sentiment(text):
    text_lower = text.lower()
    words = text_lower.split()

    pos_score = sum(1 for word in words if word in POSITIVE_WORDS)
    neg_score = sum(1 for word in words if word in NEGATIVE_WORDS)

    # Check for multi-word phrases
    for phrase in ['give up', 'falling apart', 'give up', 'not worth']:
        if phrase in text_lower:
            neg_score += 2

    if pos_score > neg_score:
        return 'Positive', '😊', 'positive'
    elif neg_score > pos_score:
        return 'Negative', '😔', 'negative'
    else:
        # Use ML model as tiebreaker
        vectorizer_local = CountVectorizer(stop_words=None, ngram_range=(1, 2))
        train_features = vectorizer_local.fit_transform(training_tweets)
        model = MultinomialNB()
        labels_shifted = [l + 1 for l in training_labels]
        model.fit(train_features, labels_shifted)
        features = vectorizer_local.transform([text])
        prediction = model.predict(features)[0] - 1
        if prediction == 1:
            return 'Positive', '😊', 'positive'
        elif prediction == -1:
            return 'Negative', '😔', 'negative'
        else:
            return 'Neutral', '😐', 'neutral'

# =====================
# ROUTES
# =====================

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('feed'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['name'] = user.name
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Invalid email or password!'})
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already exists!'})
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'error': 'Username already taken!'})
        hashed = generate_password_hash(password)
        user = User(name=name, username=username, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        session['username'] = user.username
        session['name'] = user.name
        return jsonify({'success': True})
    return render_template('signup.html')

@app.route('/feed')
def feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('feed.html',
                         username=session['username'],
                         name=session['name'])

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/posts', methods=['GET'])
def get_posts():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'})
    posts = Post.query.order_by(Post.created_at.desc()).limit(50).all()
    result = []
    for post in posts:
        liked = Like.query.filter_by(user_id=session['user_id'], post_id=post.id).first()
        comments = []
        for c in post.comments:
            comments.append({
                'id': c.id,
                'content': c.content,
                'username': c.user.username,
                'name': c.user.name,
                'created_at': c.created_at.strftime('%b %d')
            })
        result.append({
            'id': post.id,
            'content': post.content,
            'sentiment': post.sentiment,
            'sentiment_emoji': post.sentiment_emoji,
            'sentiment_color': post.sentiment_color,
            'username': post.author.username,
            'name': post.author.name,
            'created_at': post.created_at.strftime('%b %d'),
            'likes_count': len(post.likes),
            'liked': liked is not None,
            'comments': comments,
            'is_own': post.user_id == session['user_id']
        })
    return jsonify(result)

@app.route('/api/posts', methods=['POST'])
def create_post():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'})
    data = request.get_json()
    content = data.get('content', '').strip()
    if not content:
        return jsonify({'error': 'Post cannot be empty!'})
    sentiment, emoji, color = analyze_sentiment(content)
    post = Post(
        content=content,
        sentiment=sentiment,
        sentiment_emoji=emoji,
        sentiment_color=color,
        user_id=session['user_id']
    )
    db.session.add(post)
    db.session.commit()
    return jsonify({
        'success': True,
        'sentiment': sentiment,
        'sentiment_emoji': emoji,
        'sentiment_color': color
    })

@app.route('/api/like/<int:post_id>', methods=['POST'])
def toggle_like(post_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'})
    existing = Like.query.filter_by(user_id=session['user_id'], post_id=post_id).first()
    post = Post.query.get(post_id)
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({'liked': False, 'count': len(post.likes)})
    like = Like(user_id=session['user_id'], post_id=post_id)
    db.session.add(like)
    db.session.commit()
    return jsonify({'liked': True, 'count': len(post.likes)})

@app.route('/api/comment/<int:post_id>', methods=['POST'])
def add_comment(post_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'})
    data = request.get_json()
    content = data.get('content', '').strip()
    if not content:
        return jsonify({'error': 'Comment cannot be empty!'})
    comment = Comment(
        content=content,
        user_id=session['user_id'],
        post_id=post_id
    )
    db.session.add(comment)
    db.session.commit()
    return jsonify({
        'success': True,
        'comment': {
            'id': comment.id,
            'content': comment.content,
            'username': session['username'],
            'name': session['name'],
            'created_at': comment.created_at.strftime('%b %d')
        }
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)