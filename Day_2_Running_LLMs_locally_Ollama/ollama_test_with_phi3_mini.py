import ollama


message_1 = "How to run ollama using docker - Give detailed steps!"
response = ollama.chat(model='phi3:mini', 
                       messages=[{'role': 'user', 'content': message_1}])

print(response['message']['content'])