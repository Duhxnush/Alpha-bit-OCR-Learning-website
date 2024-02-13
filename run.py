from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import base64
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from keras import models
from PIL import Image as pil_image
import firestore 
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import openai
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'jaimahishmati'
openai.api_key = "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"


cred = credentials.Certificate('D:\\Alphabit website\\alpha-bit-ebc85-firebase-adminsdk-dzcr0-9c122bfd49.json')
firebase_admin.initialize_app(cred)
firestore_db = firestore.client()
db = firestore.Client.from_service_account_json('D:\\Alphabit website\\alpha-bit-ebc85-firebase-adminsdk-dzcr0-9c122bfd49.json')

# Load your pre-trained model
model = load_model('D:\\Alphabit website\\model_hand.h5')  # Update with your actual model file
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
word_dict1 = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
MODEL_PATH = "D:\\Alphabit website\\tf-cnn-model.h5"
lowmodel = load_model('Alphabet_Recognizer_32.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    username= session.get('username')
    return render_template('home.html',username=username)

@app.route('/alphabets')
def alphabets():
    return render_template('Alphabets.html')

@app.route('/numbers')
def numbers():
    return render_template('Numbers.html')

@app.route('/lowercase')
def lowercase():
    return render_template('LowerCase.html')

@app.route('/draw')
def draw():
    return render_template('draw.html')

# Start here

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Get user reference based on the 'username' field
        users_ref = db.collection('logindet')
        query = users_ref.where('username', '==', username)
        user_snapshots = query.get()

        if not user_snapshots:
            return 'User not found'

        user_data = user_snapshots[0].to_dict()

        if user_data and user_data.get('password') == password:
            session['username'] = username
            return redirect(url_for('home'))

        return 'Invalid username or password'

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

#end here

def generate_session_id():
    import uuid
    return str(uuid.uuid4())

@app.route('/predictAZ', methods=['POST'])
def predict():
    session_id = request.form.get('session_id', generate_session_id())
    # Get the base64-encoded image data from the request
    image_data = request.form['image']
    selal = request.form['selected_alphabet']
    # Decode the base64 image and save it
    img_data = image_data.split(',')[1]
    with open('temp.png', 'wb') as f:
        f.write(base64.b64decode(img_data))
        print("success")

    # Load and preprocess the image for prediction
    img = cv2.imread('temp.png')
    img_copy = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 440))

    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    img_pred = word_dict[np.argmax(model.predict(img_final))]
    confidence_scores = model.predict(img_final)[0]
    print(max(confidence_scores))
    # Convert processed image to base64 for displaying in HTML
    _, img_encoded = cv2.imencode('.png', img_copy)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    # plt.imshow(cv2.imread("temp.png"))
    # plt.show()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_to_store = {
        'model': 'Uppercase',
        'Selected':selal,
        'prediction': img_pred,
        'username': session.get('username'),  # Assuming username is stored in the session
        'timestamp': current_date,
        'confidence_score': max(confidence_scores.tolist()),
    }
    firestore_db.collection('sessions').document(session_id).set(data_to_store)


    return jsonify({
        'prediction': img_pred,
        'confidence_scores': max(confidence_scores.tolist()),
        'processedImage': img_base64,
        'session_id': session_id
    })

@app.route('/predict12', methods=['POST'])

def predict_digit():
    session_id = request.form.get('session_id', generate_session_id())
    # load model
    
    
    model = models.load_model(MODEL_PATH)
    print("[INFO] Loaded model from disk.")

    image_data = request.form['image']
    selal = request.form['selected_alphabet']
    # Decode the base64 image and save it
    img_data = image_data.split(',')[1]
    with open('temp.png', 'wb') as f:
        f.write(base64.b64decode(img_data))
        print("success")
    image = cv2.imread('temp.png')   
    img_copy=image   
    img = cv2.imread('temp.png')
    img_copy = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 440))
    img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))

    img_pred = word_dict1[np.argmax(model.predict(img_final))]
    confidence_scores = model.predict(img_final)[0]

    # Convert processed image to base64 for displaying in HTML
    _, img_encoded = cv2.imencode('.png', img_copy)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    # plt.imshow(cv2.imread("temp.png"))
    # plt.show()
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_to_store = {
        'model': 'Numbers',
        'Selected':selal,
        'prediction': img_pred,
        'username': session.get('username'),  # Assuming username is stored in the session
        'timestamp': current_date,
        'confidence_score': max(confidence_scores.tolist()),
    }
    firestore_db.collection('sessions').document(session_id).set(data_to_store)

    return jsonify({
        'prediction': img_pred,
        'confidence_scores': max(confidence_scores.tolist()),
        'processedImage': img_base64,
    })   



@app.route('/predictab', methods=['POST'])
def predict_lowercase():
    
    session_id = request.form.get('session_id', generate_session_id())
    image_data = request.form['image']
    selal = request.form['selected_alphabet']
    # Decode the base64 image and save it
    img_data = image_data.split(',')[1]
    with open('temp.png', 'wb') as f:
        f.write(base64.b64decode(img_data))
        print("success")
    image_path = 'temp.png'
    
    # Read the image in color
    img_color = cv2.imread(image_path)
    
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 28x28
    img_gray_resized = cv2.resize(img_gray, (28, 28))
    
    # Maximum pixel value
    max_val = 255
    
    complement_img = pil_image.fromarray(max_val - img_gray_resized)
    img = np.array(complement_img)
    img = img.reshape((1, 28, 28, 1))
    img = img.astype('float32') / 255.0
    
    # Make prediction (mocking the model for demonstration)
    pred = lowmodel(img)
    y_pred = np.argmax(pred, axis=1)
    
    # Get the predicted character
    predicted_char = chr(y_pred[0] + 97)
    
    confidence_score = lowmodel.predict(img)[0]
    print(confidence_score)
    # Get the processed image as a NumPy array
    pred_image = np.squeeze(img)
    
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_to_store = {
        'model': 'Lowercase',
        'Selected':selal,
        'prediction': predicted_char,
        'username': session.get('username'),  # Assuming username is stored in the session
        'timestamp': current_date,
        'confidence_score': max(confidence_score.tolist()),
    }
    firestore_db.collection('sessions').document(session_id).set(data_to_store)
    
    return jsonify({
        'prediction': predicted_char,
        'confidence_scores': max(confidence_score.tolist()) ,
        'processedImage': pred_image.tolist(),  # Convert ndarray to list
    })

@app.route('/get_insights/<session_id>')
def get_insights(session_id):
    session_ref = firestore_db.collection('sessions').document(session_id)
    session_data = session_ref.get().to_dict()

    if not session_data:
        return jsonify({'error': 'Session not found'})

    # Process session data to generate insights (replace this with your analysis logic)
    insights = {
        'total_predictions': 1,
        'models_used': [session_data['model']],
        'session_id': session_id,
        # Add more insights based on your requirements
    }

    # Clear the session data after retrieving insights
    session_ref.delete()

    return jsonify(insights)

@app.route('/user_insights/<username>', methods=['GET'])
def user_insights(username):
    # Query Firestore for user details for the specified username and current date
    user_sessions = firestore_db.collection('sessions').where('username', '==', username).stream()
    
    # Prepare messages for the chat-based model
    messages = [
        {"role": "system", "content": "Analyze the provided dataset on a child's alphabet and number tracing practice sessions. Discuss the frequency of practicing the same character, evaluate how closely the child is learning specific alphabets, and provide recommendations on how the child can improve. Consider the repetition of characters, confidence scores, and suggest strategies to enhance the learning experience, making the response around 300 words."},
    ]

    for session in user_sessions:
        session_data = session.to_dict()
        # Add relevant session details to chat input (customize as needed)
        print(session_data['prediction'])
        messages.append({"role": "user", "content": f"Model: {session_data['model']}, Prediction: {session_data['prediction']}, Confidence: {session_data['confidence_score']}"})

    # Use the ChatGPT API to generate insights
    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the appropriate ChatGPT model
        messages=messages,
        max_tokens=1000,  # Adjust as needed
        temperature=0.7  # Adjust as needed
    )

    # Extract insights from ChatGPT response
    chatgpt_insights = chatgpt_response['choices'][0]['message']['content']

    # Render HTML page with the generated insights
    return render_template('insights.html', username=username, insights=chatgpt_insights)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists in the collection
        users_ref = firestore_db.collection('logindet')
        query = users_ref.where('username', '==', username)
        user_snapshots = query.get()

        if user_snapshots:
            return 'Username already exists. Please choose another username.'

        # Create a new document in the "logindet" collection
        new_user_ref = users_ref.document()
        new_user_ref.set({
            'username': username,
            'password': password,  # Note: Passwords should be securely hashed in a real-world application
            # Add other user-related data if needed
        })

        return f'Successfully signed up as {username}!'

    return render_template('signup.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
