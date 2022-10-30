import torch 
# import numpy as np
from mingpt.utils import set_seed 
from mingpt.model import GPT
from mingpt.trainer import Trainer
from LanguageModelingDataset import LanguageModelingDataset
from utils import *

set_seed(1008498261)
torch.manual_seed(42)
# np.random.seed(42)

if __name__ == "__main__":
    print(f'cuda=={torch.cuda.is_available()}')
    
    # Instantiate the Training Dataset on Large Corpus
    # train_dataset = LanguageModelingDataset(ds_choice="small", split="train")  # use this for the short corpus
    train_dataset = LanguageModelingDataset(ds_choice="large", split="train", truncation=512) #use this for long

    # Instantiate a Validation Dataset (this is only really needed for the fine-tune task, not the LM task)
    # val_dataset = LanguageModelingDataset(ds_choice="small", split="validation")
    val_dataset = LanguageModelingDataset(ds_choice="large", split="validation", truncation=512)
    
    
    # Create and config model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model_config.n_classification_class = 2
    model = GPT(model_config)
    
    # Load state dict
    modelsavename= "model_filename.pt"
    model.load_state_dict(torch.load(modelsavename))
    
    # Create a Trainer object and set the core hyper-parameters
    train_config = Trainer.get_default_config()
    trainer = Trainer(train_config, model, train_dataset, val_dataset, collate_fn=lm_collate_fn)
    
    # Test the results
    encoded_prompt = train_dataset.tokenizer("Having refered to").to(trainer.device)
    generated_sequence = trainer.model.generate(encoded_prompt, trainer.device, temperature=0.8, max_new_tokens=10)
    
    result = train_dataset.tokenizer.decode(generated_sequence[0])

    print(result)