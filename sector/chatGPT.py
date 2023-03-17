import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def send(content, role='user', max_tokens=500):
    if content == '' or content is None:
        return None
    else:
        return openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                max_tokens=max_tokens,
                messages=[{'role': role, 'content': content}],
        )       


############

# 构造能够直接发送的请求
def create_request_content(ld):
    requests = []
    for ss, labels in ld:
        text = ''
        for idx, s in enumerate(ss):
            if idx == 2:
                s = '🔪' + s
            if s is not None:
                text += s.strip()
        requests.append(text)
    requests = [f'Please judge whether the marked sentence is the beginning of a paragraph: {text}' for text in requests]
    return requests

def create_request_content_cot(ld):
    requests = []
    for ss, labels in ld:
        text = ''
        for idx, s in enumerate(ss):
            if idx == 2:
                s = '🔪' + s
            if s is not None:
                text += s.strip()
        requests.append(text)
    requests = [f"Q: Is the marked sentence the beginning of a new paragraph? Context: {text} \n A: Let's think step by step then answer yes or no." for text in requests]
    return requests

def script():
    print('Please take a look in ./chatgpt/README.md')


# @return: label, evidence
def transfer_response_to_label(text):
    text = text.lower().strip()
    if text.find('yes') != -1:
        if text.find('no') != -1:
            print(f'E0, CAN NOT TRANSFER: {text}')
            return None, text
        else:
            temp = text.find('yes')
            return 1, text[temp - 10: temp + 10]
    elif text.find('no') != -1:
        if text.find('yes') != -1:
            print(f'E1, CAN NOT TRANSFER: {text}')
            return None, text
        else:
            temp = text.find('no')
            return 0, text[temp - 10: temp + 10]
    else:
        print(f'E2, CAN NOT IDENTIFY: {text}')
        return None, text





