from dotmap import DotMap
import torch
import torchtext
import gradio as gr
from A2_CNN.Classifier import Conv2dWordClassifier

glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100

# path1 = "A2_CNN\models\model_cnn_10112022_174856.pt"
# args1 = DotMap({'k1': 2, 'k2': 4, 'n1': 4, 'n2': 16, 'freeze_embedding': False, 'bias': False})
path2 = "A2_CNN\models\model_cnn_10112022_174856.pt"
args2 = DotMap({'k1': 2, 'k2': 5, 'n1': 16, 'n2': 32, 'freeze_embedding': False, 'bias': False})

def predict(s):
    m = Conv2dWordClassifier(glove, args2)
    m.load_state_dict(torch.load(path2))
    m.eval()
    
    tokens = s.split()
    token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]
    token_tensor = torch.LongTensor(token_ints).view(-1,1)
    _, logit = m(token_tensor)
    pred = torch.round(logit).long()
    return "Subjective" if pred else "Objective"


if __name__ == "__main__":

    # m1 = Conv2dWordClassifier(glove, args1)
    # m2 = Conv2dWordClassifier(glove, args2)
    
    gApp = gr.Interface(fn=predict, inputs="text", outputs="text")

    gApp.launch()