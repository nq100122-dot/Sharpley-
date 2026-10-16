""" script_train.py

	Train, evaluate and save the desired GNN model 
	on the given dataset. 
"""

import argparse
import os
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import configs
from src.data import prepare_data
from src.models import GAT, GCN, GcnEncoderGraph, GcnEncoderNode, GCNNet
from src.train import evaluate, train_and_val, train_gc, train_syn
from src.utils import *
from utils.io_utils import fix_seed


def main():

    args = configs.arg_parse() 
    fix_seed(args.seed)

    # Load the dataset
    #Nạp dữ liệu và chia train/val/test
    data = prepare_data(args.dataset, args.train_ratio, args.input_dim, args.seed)

    # Define and train the model
    if args.dataset in ['Cora', 'PubMed']:
        # Retrieve the model and training hyperparameters depending the data/model given as input # Lấy thông số của mô hình
        #Nhánh if: Cora và PubMed, dùng GCN hoặc GAT với hàm train train_and_val
        hyperparam = ''.join(['hparams_', args.dataset, '_', args.model]) # Lấy tên file hyperparameter
        param = ''.join(['params_', args.dataset, '_', args.model]) # Lấy tên file tham số
        model = eval(args.model)(input_dim=data.num_features, output_dim=data.num_classes, **eval(hyperparam)) # Khởi tạo mô hình
        train_and_val(model, data, **eval(param)) # Huấn luyện mô hình
        _, test_acc = evaluate(data, model, data.test_mask) # Đánh giá mô hình
        print('Test accuracy is {:.4f}'.format(test_acc)) # In ra độ chính xác
#Nhánh elif: syn6 và Mutagenicity, dùng GcnEncoderGraph với hàm train train_gc.
    elif args.dataset in ['syn6', 'Mutagenicity']:
        input_dims = data.x.shape[-1]
        model = GcnEncoderGraph(input_dims,
                            args.hidden_dim,
                            args.output_dim,
                            data.num_classes,
                            args.num_gc_layers,
                            bn=args.bn,
                            dropout=args.dropout,
                            args=args)
        train_gc(data, model, args) # Huấn luyện mô hình
        _, test_acc = evaluate(data, model, data.test_mask) # Đánh giá mô hình
        print('Test accuracy is {:.4f}'.format(test_acc)) # In ra độ chính xác
#Nhánh else: syn1, syn2, syn4 và syn5, dùng GcnEncoderNode với hàm train train_syn.
    else: 
        # For pytorch geometric model # Dùng GNN của pytorch geometric
        #model = GCNNet(args.input_dim, args.hidden_dim,
        #       data.num_classes, args.num_gc_layers, args=args) 
        input_dims = data.x.shape[-1] # Lấy số chiều của input
        model = GcnEncoderNode(data.num_features,
                                args.hidden_dim,
                                args.output_dim,
                                data.num_classes,
                                args.num_gc_layers,
                                bn=args.bn,
                                dropout=args.dropout,
                                args=args)
        train_syn(data, model, args) # Huấn luyện mô hình
        _, test_acc = evaluate(data, model, data.test_mask) # Đánh giá mô hình
        print('Test accuracy is {:.4f}'.format(test_acc)) # In ra độ chính xác
    
    # Save model # Lưu mô hình 
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset) # Lưu mô hình
    if not os.path.exists(model_path) or args.save==True:
        torch.save(model, model_path) # Lưu mô hình vào file


if __name__ == "__main__":
    main()
