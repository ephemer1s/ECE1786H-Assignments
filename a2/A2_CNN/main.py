import argparse


try:
    from utils import *
    from train import *
except Exception as e: 
    print(e)
    print('trying another import path')
    from A2_CNN.utils import *
    from A2_CNN.train import *
    print('import successful')
    
# 4.3 parse args
parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch_size", type=int, default=64)
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument("-ml", "--max_len", type=int, default=0)
parser.add_argument("-s", "--save_model", type=bool, default=False)
parser.add_argument("-o", "--overfit_debug", type=bool, default=False)
parser.add_argument("-b", "--bias", type=bool, default=False)

# added in 5.2
parser.add_argument("-k1", "--k1", type=int, default=2)
parser.add_argument("-n1", "--n1", type=int, default=10)
parser.add_argument("-k2", "--k2", type=int, default=4)
parser.add_argument("-n2", "--n2", type=int, default=10)
parser.add_argument("-f", "--freeze_embedding", type=bool, default=True)
parser.add_argument("-g", "--grid_search", type=bool, default=False)
parser.add_argument("-v", "--val_epoch", type=int, default=2)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.grid_search:
        print("Using Grid Search")
        
        best_model, best_result, best_args = grid_search()
        
        save_results(best_result, best_args)
        
    else:
        
        train_dataloader, validation_dataloader, test_dataloader = preprocess(args)
        model, train_loss, train_acc, val_loss, val_acc, test_acc = train(args, train_dataloader, validation_dataloader, test_dataloader)
            
        # save results
        save_results([train_loss, train_acc, val_loss, val_acc, test_acc], args)

        if args.save_model:
            save_model(model)
