import json
import requests
import numpy as np
from rnn_model import my_tokenizer
if __name__ == '__main__':

    sentences = my_tokenizer.texts_to_sequences([" ".join("今天下大雪，外卖小哥没有迟到，真的敬业，点个赞！")])

    payload = {
        "instances": [{'input_sentence': sentences[0]}]
    }

    print(requests.post('http://127.0.0.1:9000/v1/models/ec:predict', json=payload).content)
