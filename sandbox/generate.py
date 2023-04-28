import openai

def generate(prompt:str, input:str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":prompt+input}])
    code = response['choices'][0]['message']['content']
    return code

if __name__ == "__main__":
    openai.debug = True
    print(generate(open("test_prompt.txt").read(), open("test_input.txt").read()))
