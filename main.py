import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [{
        "key": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Yes?", "Hello", "Hey", "Hello to you 2", "What?"]
    },
    {
        "key": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye to you too", "OK", "Take care"]
    },
    {
        "key": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["OK", "No problem", "You are absolutely welcome"]
    },
    {
        "key": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose", "What are you able to do"],
        "responses": ["I am Edbert's personal information chatbot", "My purpose is to provide information about Edbert", "I can answer questions about Edbert"]
    },
    {
        "key": "who",
        "patterns": ["Who", "Who is Edbert", "Who is he", "Who is that", "What is an Edbert"],
        "responses": ["Edbert is a second year Data Science Student at the University of Hong Kong.", "He's my owner. He's a university student deeply interested in AI and finance", "He's an indonesian studying Data Science at HKU. He's currently exploring the world of AI and that's why he made me."]
    },
    {
        "key": "age",
        "patterns": ["How old are you", "What's your age", "age"],
        "responses": ["I'm about an month old", "I'm still a baby born in the 21st century", "I'm young but I'm competent", "I'm older than you", "You have no business knowing that"]
    },
    {
        "key": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I told you I only provide information about Edbert", "I don't know. Ask someone else please"]
    },
    {
        "key": "location",
        "patterns": ["Where is Edbert now", "Where is he", "What's Edbert's location right now","What is Edbert's whereabouts right now", "What is his location"],
        "responses": ["He's currently at Jakarta, Indonesia enjoying his summer break. Feel free to contact him at whatsapp +852 9324 2257 or personal email edbertwid88@gmail.com"]
    },
    {
        "key": "hobbies",
        "patterns": ["What are Edbert's hobbies", "What are his interests", "What is Edbert interested in"],
        "responses": ["Edbert's a fan of various things. He likes trying new things and food. In his free time, he hits the gym", "Edbert has a keen interest in the world of AI and Data Science. He likes exploring new aspects of the field that are previously unknown to him."]
    },
    {
        "key": "personality",
        "patterns": ["What's Edbert like","What's his personality type","How is he with others"],
        "responses": ["Why don't you contact him and find out"]
    },
    {
        "key": "free",
        "patterns": ["When is he free", "When is he available", "When is his free time"],
        "responses":["Anytime is good. Just contact him", "Please contact him"]
    }
]
# Should Add more Intents for better Bot functionality

vectorizer = TfidfVectorizer()
model = LogisticRegression(random_state=0, max_iter=10000)

keys = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        keys.append(intent["key"])
        patterns.append(pattern)

X = vectorizer.fit_transform(patterns)
y = keys
model.fit(X, y)

def chat(text):
    text = vectorizer.transform([text])
    key = model.predict(text)[0]
    for intent in intents:
        if intent['key'] == key:
            response = random.choice(intent['responses'])
            return response

# USING STREAMLIT FOR USER INTERFACE
count = 0
def main():
    global count
    st.title("Edbert's Personal Info Chatbot")
    st.write("Welcome. Please type a message and press Enter to start the conversation.")

    count += 1
    user_input = st.text_input("You:", key=f"user_input_{count}")

    if user_input:
        response = chat(user_input)
        st.text_area("Edbert's Bot:", value=response, height=101, max_chars=None, key=f"chatbot_response_{count}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()

# USING DASH FRAMEWORK FOR USER INTERFACE
history = []
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Edbert's Chatbot",style={'text-align': 'center'}),
    
    html.Div([
        html.Div([
            html.Table([
                html.Tr([
                    html.Td([dcc.Input(id = "messageinput", value = "hello", type = "text")],
                           style = {'valign': "middle"}),
                    html.Td([html.Button("Send", id = "sendButton", type="submit")],
                           style = {'valign': "middle"})
                ])
            ])
        ], style = {"width": "330px", "margin": "0 auto"}),
        html.Br(),
        html.Div(id = "convo")],
        id = "screen",
        style = {"width": "400px", "margin":"0 auto"})
        
])

@app.callback(
    Output(component_id = "convo", component_property = "children"),
    [Input(component_id = "sendButton", component_property = "n_clicks")],
    [State(component_id = "messageinput", component_property = "value")]
)

def update_output(click,text):
    global history
    if click > 0:
        response = chat(text)
        user = [html.H5(text, style = {"text-align": "left"})]
        bot = [html.H5(html.I(i), style = {"text-align": "right"}) for i in response]
        history = user + bot + [html.Hr()] + history
        return history
    else:
        return ''
    
@app.callback(
    Output(component_id = "messageinput", component_property = "value"),
    [Input(component_id = "convo", component_property = "children")]
)

def clearInput(_):
    return ''

if __name__ == "__main__":
    app.run_server(debug = True)
