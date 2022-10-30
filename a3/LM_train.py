import torch 
# import numpy as np
from mingpt.utils import set_seed 
from mingpt.model import GPT
from mingpt.trainer import Trainer
from LanguageModelingDataset import LanguageModelingDataset
from utils import *

set_seed(42)
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


    # Create a Trainer object and set the core hyper-parameters
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 100000  # For small corpus: 3000 iterations is plenty. For large corpus: 100000 iterations is needed
    train_config.num_workers = 0
    train_config.batch_size = 8    # For small corpus, batch size of 4 is fine.  For large corpus use 16
    trainer = Trainer(train_config, model, train_dataset, val_dataset, collate_fn=lm_collate_fn)


    trainer.set_callback('on_batch_end', batch_end_callback)

    # Train!
    trainer.run()

    model.to(trainer.device)
    # store the saved model in a file, so can re-use later
    modelsavename= "model_filename.pt"  # change the name here to save in a specific file (and restore below)
    with open(modelsavename, "wb") as f:
        torch.save(trainer.model.state_dict(), f)
        
