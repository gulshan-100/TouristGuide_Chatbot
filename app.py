# app.py

from flask import Flask, request, render_template, jsonify, send_from_directory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv(find_dotenv())

app = Flask(__name__)

# Initialize the ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model='gemini-pro',
    temperature=0.5
)

groq_model = ChatGroq(
    model = 'llama3-70b-8192'
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('templates', 'styles.css')

@app.route('/results', methods=['POST'])
def predict():
    city = request.form.get('city')

    template = """Explain  the importance of this city. 
    Suggest 5 places or tourist destinations to visit in this city. The response should be in proper format with spaces between lines.
    If you don't know the answer, you can reply with "I don't know". Don't respond with misleading information.
    
    City : {city}
    Answer: """

    prompt = PromptTemplate(
        input_variables=['city'],
        template=template
        
    )

    chain = LLMChain(
        llm=model,
        prompt=prompt
    )

    output = chain.run(city)

    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)