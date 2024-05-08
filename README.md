https://www.youtube.com/watch?v=UJNlPA83IJY&t=985s

To run the code install the dependencies given in requirements.txt	
Preferred way is to create an virtual env and proceed.
We have used UBUNTU, for other OS addition dependencies or versions might be required.

To run the code as is,

1. install all dependencies in a virtual environment
2. activate the environment
3. Also install Ollama for Local LLMs (refer official website) [curl -fsSL https://ollama.com/install.sh | sh] UBUNTU
4. download llama2 in Ollama
5. now 'cd NER'
6. run all the NER_Training.ipynb, it will create a trained ner model which can be used further.
7. 'cd ..'
8. now run the Final.ipynb file to produce the shown results.

To access through the Frondend, 
run

'streamlit run app.py'

 
