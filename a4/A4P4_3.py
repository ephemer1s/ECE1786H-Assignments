from treelib import Node, Tree
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "It is important for all countries to try harder to reduce carbon emissions because"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask

torch.manual_seed(0)
tree = Tree()
tree.create_node(tag='because', identifier='root')  # root

def nextword(input_ids):
    tree = Tree()
    depth = 4 # as 3 ** 5 == 243 nodes will be not so practical to visualize
    cur_length = len(input_ids[0])
    # obtain next pred
    outputs = model.generate(
        input_ids, 
        attention_mask=torch.ones((1, cur_length)), 
        pad_token_id=50256,
        do_sample=True, 
        max_length=cur_length+1,
        temperature=0.5, 
        top_p=0.95,
        return_dict_in_generate=True,
        output_scores=True,)
    word = []
    decode = []
    # decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    # print(decoded)
    probs = outputs.scores[-1][0]
    val, idx = torch.topk(probs, 3)
    for i, v in zip(idx, val):
        word.append(i.item())
        # decode.append(tokenizer.decode(i))
    # print(decode)
    return word


d1 = nextword(input_ids)
print(d1)

d2 = []
for i in d1:
    tmp = input_ids[0].tolist()
    tmp.append(i)
    d2.append(nextword(torch.tensor([tmp])))
print(d2)

d3 = []
for i in range(3):
    for j in range(3):
        tmp = input_ids[0].tolist()
        tmp.append(d1[i])
        tmp.append(d2[i][j])
        d3.append(nextword(torch.tensor([tmp])))
d3 = np.array(d3).reshape(3,3,3).tolist()
print(d3)

d4 = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            tmp = input_ids[0].tolist()
            tmp.append(d1[i])
            tmp.append(d2[i][j])
            tmp.append(d3[i][j][k])
            d4.append(nextword(torch.tensor([tmp])))
d4 = np.array(d4).reshape(3,3,3,3).tolist()
print(d4)

for i in range(3):
    tree.create_node(tag=tokenizer.decode(d1[i]), identifier=i, parent='root')
    for j in range(3):
        i2 = str(i) + str(j)
        tree.create_node(tag=tokenizer.decode(d2[i][j]), identifier=i2, parent=i)
        for k in range(3):
            i3 = str(i) + str(j) + str(k)
            tree.create_node(tag=tokenizer.decode(d3[i][j][k]), identifier=i3, parent=i2)
            for l in range(3):
                tree.create_node(tag=tokenizer.decode(d4[i][j][k][l]), parent=i3)

tree.show()
