import os
import torch
import argparse
import numpy as np

from Models import AdaFilterModel, CTLSTM, GTPP
from Trainers import AdaFilterTrainer
from Utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="mooc", type=str)
    parser.add_argument("--do_eval",default=False, type=bool)
    parser.add_argument("--load_model", default=None, type=str)

    # model args
    parser.add_argument("--model_name", default="AdaFilter", type=str)
    parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
    parser.add_argument("--barrel_size", default=100, type=int, help="separated barrel size of time")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_attention_heads", default=4, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.0, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.0, type=float)
    parser.add_argument("--initializer_range", default=0.001, type=float)
    parser.add_argument("--max_seq_length", default=0, type=int, help="max length of processed sequences, 0 means selecting according to the trainset")

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=64, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")
    parser.add_argument("--local", default=1, type=int)
    parser.add_argument("--pool_size", default=3, type=int)
    parser.add_argument("--use_layernorm", default=True, type=bool)
    parser.add_argument("--use_residual", default=True, type=bool)
    parser.add_argument("--inner_size", default=128, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    parser.add_argument("--data_type", default="sequential_based", type=str)
    parser.add_argument("--benchmark_type", default="hybrid_retrieval", type=str)
    parser.add_argument("--use_baseline", default=False, type=bool)
    parser.add_argument("--use_metaformer", default=False, type=bool)
    parser.add_argument("--ffn_multiplier", default=4, type=int)
    parser.add_argument("--temporal_encoding_method", default="LTR", type=str)
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item = get_seq_dic(args)

    # save model args
    cur_time = get_local_time()
    args_str = f'{args.model_name}-{args.data_name}-{cur_time}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)
    if args.model_name == "NHP":
        model = CTLSTM(args=args)
    elif args.model_name == "FullyNN":
        model = GTPP(args=args)
    else:    
        model = AdaFilterModel(args=args)
    trainer = AdaFilterTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("---------------Sample 99 results---------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

main()

