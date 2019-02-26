import json
import requests
import numpy as np

# payload ={
#     "instances": [{'input_sentence': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}]
# }
#
# print(requests.post('http://127.0.0.1:9000/v1/models/mymodel:predict', json=payload).content)
f = open("train.tsv", 'r', encoding="utf-8")
for i in f.readlines():
    print(i.split("\t"))