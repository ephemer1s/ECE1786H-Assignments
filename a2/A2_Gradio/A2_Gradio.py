from dotmap import DotMap
import torch
import torchtext
import gradio as gr
try:
    from Classifier import Conv2dWordClassifier
except Exception as e:
    print(e)
    from A2_Gradio.Classifier import Conv2dWordClassifier


glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100

path1 = "A2_Gradio/models/model_cnn_10112022_175513.pt"
args1 = DotMap({'k1': 2, 'k2': 4, 'n1': 4, 'n2': 16, 'freeze_embedding': False, 'bias': False})
path2 = "A2_Gradio/models/model_cnn_10112022_174856.pt"
args2 = DotMap({'k1': 2, 'k2': 5, 'n1': 16, 'n2': 32, 'freeze_embedding': False, 'bias': False})

def predict(sentence, model):
    if model == "model 2":
        m = Conv2dWordClassifier(glove, args2)
        m.load_state_dict(torch.load(path2))
    else:
        m = Conv2dWordClassifier(glove, args1)
        m.load_state_dict(torch.load(path1))
    m.eval()
    tokens = sentence.split()
    token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]
    token_tensor = torch.LongTensor(token_ints).view(-1,1)
    _, logit = m(token_tensor)
    pred = torch.round(logit).long()
    return "Subjective" if pred else "Objective"


if __name__ == "__main__":

    # m1 = Conv2dWordClassifier(glove, args1)
    # m2 = Conv2dWordClassifier(glove, args2)
    
    gApp = gr.Interface(fn=predict, 
                        inputs=[gr.Textbox(label="type a sentence here"), 
                                gr.Radio(["model 1", "model 2"])], 
                        outputs=gr.Textbox(label="Model's Prediction"))

    gApp.launch()