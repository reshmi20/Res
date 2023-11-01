<h1>Create a Chatbot Using Python</h1>

<h2>Introduction:</h2>

<p>This repository contains Python code for a customer service chatbot implemented using Flask, SpaCy, and the GPT-2 language model. The chatbot is designed to provide high-quality support to users, answering their queries on a website or application.</p>

<h2>Dependencies:</h2>

<p>Ensure you have the following dependencies installed in your Python environment:</p>

<ul>
  <li>Python 3.x</li>
  <li>Flask</li>
  <li>SpaCy</li>
  <li>Pandas</li>
  <li>Transformers (from Hugging Face)</li>
</ul>

<p>You can install the required packages using the following command:</p>

<pre><code>bash: pip install flask spacy pandas transformers</code></pre>

<p>Additionally, download the SpaCy model using:</p>

<pre><code>bash: python -m spacy download en_core_web_sm</code></pre>

<p>Download the GPT-2 model using the following Python code:</p>

<pre><code>python:
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
</code></pre>

<h2>Setting Up the Environment</h2>

<p>Create a Virtual Environment: It's a good practice to work within a virtual environment to manage project dependencies. To create a virtual environment, run the following commands in your project directory:</p>

<pre><code>bash: python3 -m venv venv</code></pre>

<p>Activate the Virtual Environment On Windows:</p>

<pre><code>bash: venv\Scripts\activate</code></pre>

# Template:index.html

    <!DOCTYPE html>
    <html lang="en">

    <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
    <style>
        body {
            display: flex;
            background-color: #007bff;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .container {
            
            max-width: 400px;
            width: 100%;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .message {
            margin-bottom: 10px;
        }

        .user-message strong {
            color: #007bff;
        }

        .bot-message strong {
            color: #28a745;
        }
    </style>
    </head>

    <body>
    
    <div class="container">
        <h1 style="text-align: center;">Chatbot</h1>
        <div class="message user-message">
            <strong>You:</strong> {{ user_input }}
        </div>
        <div class="message bot-message">
            <strong>Bot:</strong> {{ bot_response }}
        </div>
        <form method="POST" action="/chat" style="text-align: center;">
            <label for="user_input">You:</label>
            <input type="text" id="user_input" name="user_input" value="{{ user_input }}">
            <input type="submit" value="Ask">
        </form>
    </div>
    </body>
    </html>
# dataset.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Preprocessed Dataset</title>
</head>
<body>
    <h1>Preprocessed Dataset</h1>
    <table border="1">
        <thead>
            <tr>
                <th>Question</th>
                <th>Answer</th>
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
                <tr>
                    <td>{{ row.question }}</td>
                    <td>{{ row.answer }}</td>
                </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>





# CHATBOT.py
    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Flask setup
    app = Flask(_name_)

    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load the dataset from the specified file path
    dataset = pd.read_csv('dialogs.txt', delimiter="\t", header=None, names=["question", "answer"])

    # Define the clean_text function to preprocess text data
    def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

    # Define the remove_repeating_sentences function to remove repeating sentences from a dataset
    def remove_repeating_sentences(dataset):
    seen_sentences = set()
    filtered_dataset = []

    for index, row in dataset.iterrows():
        if row["question"] not in seen_sentences:
            seen_sentences.add(row["question"])
            filtered_dataset.append(row)

    return pd.DataFrame(filtered_dataset)

    # Preprocess the dataset
    dataset = dataset.dropna()
    dataset["question"] = dataset["question"].apply(clean_text)
    dataset["answer"] = dataset["answer"].apply(clean_text)
    dataset = remove_repeating_sentences(dataset)

    #flask

    # Flask route for chatbot and dataset
    @app.route('/')
    def index():
    return render_template('index.html')

    @app.route('/chat', methods=['POST'])
    def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_input = clean_text(user_input)

        # Check if the user input matches any question in the preprocessed dataset
        matching_row = dataset[dataset['question'] == user_input]
        
        if not matching_row.empty:
            # If a matching question is found, retrieve the corresponding answer
            bot_response = matching_row['answer'].values[0]
        else:
            # If no matching question is found, generate a response using the GPT-2 model
            input_ids = tokenizer.encode(user_input, return_tensors='pt')
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
            bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return render_template('index.html', user_input=user_input, bot_response=bot_response)
    return render_template('index.html')

    @app.route('/dataset')
    def show_dataset():
    return render_template('dataset.html', data=dataset.to_dict(orient='records'))

    if _name_ == '_main_':
    app.run(debug=True)

<h2>How to Run:</h2>

<p>Place your dataset file dialogs.txt in the root directory. The dataset should be in tab-separated format with columns "question" and "answer".</p>
<p>Dataset: <a href="https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot">https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot</a></p>

<p>Run the Flask application:</p>

<pre><code>bash: python app.py</code></pre>

<p>Access the chatbot interface in your web browser at <a href="http://localhost:5000">http://localhost:5000</a>.</p>

<h2>Dataset</h2>

<p>The dataset (dialogs.txt) used for this chatbot contains pairs of questions and corresponding answers. It's utilized to train the chatbot and provide predefined responses. Please replace provide_source_link_here with the actual source link of your dataset.</p>
<p>Dataset: <a href="https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot">https://www.kaggle.com/datasets/grafstor/simple-dialogs-for-chatbot</a></p>


<h2>Usage</h2>

<p>Users can input their queries through the chat interface. If the query matches a question in the dataset, the corresponding answer is provided. If there is no match in the dataset, the chatbot uses the GPT-2 model to generate a response.</p>

<h2>Additional Notes</h2>

<p>The chatbot's responses can be further enhanced by integrating advanced models like GPT-3. Implement feedback mechanisms to continuously improve response quality.</p>

<p><strong>Note:</strong> Ensure that you have the necessary API keys and permissions if you plan to use advanced language models like GPT-3.</p>
