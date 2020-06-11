import argparse
import os
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_pretrained_bert import BertModel

from torchvision import transforms
from torch.optim import RMSprop

from utils.text_process import process_dataframe
from utils.models import ImageTextDataset, Towers, return_adj_matrix
from utils.util import get_encoding, classification_loss_fn, graph_loss_fn, gap_loss_fn

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Hierarchical Universal Semantic Embedding")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=100,
                                  help="number of training epochs, default is 100")
    train_arg_parser.add_argument("--batch_size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--csv_file_path", type=str, default= './data/train_data.csv',
                                  help="path to the csv file which contains the train image paths and text, default is './data/train_data.csv'")
    train_arg_parser.add_argument("--classification_weight", type=float, default=1,
                                  help="weight for classification-loss, default is 1")
    train_arg_parser.add_argument("--graph_weight", type=float, default=10,
                                  help="weight for graph-loss, default is 10")
    train_arg_parser.add_argument("--gap_weight", type=float, default=4,
                                  help="weight for gap-loss, default is 4")                                  
    train_arg_parser.add_argument("--margin", type=float, default=0.7,
                                  help="margin for graph loss, default is 0.7")                                  
    train_arg_parser.add_argument("--lr", type=float, default=1.6192e-05,
                                  help="learning rate, default is 1.6192e-05")
    train_arg_parser.add_argument("--save_model_dir", type=str, default= './saved_models/',
                                   help="path to folder where trained model will be saved, default is ./saved_models/")
    train_arg_parser.add_argument("--checkpoint_model_dir", type=str, default='./checkpoints/',
                                  help="path to folder where checkpoints of trained models will be saved, default is ./checkpoints/")
    train_arg_parser.add_argument("--cuda", type=int, default = 1,
                                  help="set it to 1 for running on GPU, 0 for CPU, default is 1")
    train_arg_parser.add_argument("--log_interval", type=int, default=100,
                                  help="number of images after which the training loss is logged, default is 100")
    train_arg_parser.add_argument("--checkpoint_interval", type=int, default=1,
                                  help="number of epochs after which a checkpoint of the trained model will be created, default is 1")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation")
    eval_arg_parser.add_argument("--csv_file_path", type=str, default= './data/eval_data.csv',
                                  help="path to the csv file which contains the validation image paths and text")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                  help="saved model to be used for evaluating the images")
    eval_arg_parser.add_argument("--cuda", type=int, default=1,
                                 help="set it to 1 for running on GPU, 0 for CPU, default is 1")
    args = main_arg_parser.parse_args()
    
    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        evaluate(args)

def train(args):

    device = torch.device("cuda:0" if args.cuda else 'cpu')

    original_dataframe = pd.read_csv(args.csv_file_path)
    processed_dataframe = process_dataframe(original_dataframe)
    adj_matrix = return_adj_matrix(processed_dataframe)
    X = processed_dataframe.loc[:, ['image', 'processed_text']].values
    y = processed_dataframe['mapped_classes'].values
    
    transform = transforms.Compose([transforms.RandomRotation(5),
                                    transforms.RandomResizedCrop(284, scale = (0.9, 1.0)),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.1, hue=0.07), # Adding some random jitter
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
    trainset = ImageTextDataset(X, y, transform)
    trainloader = torch.utils.data.DataLoader(trainset, shuffle = True, batch_size = args.batch_size, drop_last= True)

    resnet50 = models.resnet50(pretrained = True).to(device)
    resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1])).to(device)
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)

    for param in resnet50.parameters():
        param.requires_grad = False

    for param in bert.parameters():
        param.requires_grad = False

    model = Towers(len(np.unique(y))).to(device)
    opt = RMSprop(model.parameters(), lr = args.lr, momentum = 0.9)

    for e in range(args.epochs):
        loss_agg =  0
        classification_loss_agg = 0
        graph_loss_agg = 0
        gap_loss_agg = 0

        for batch_id, data in enumerate(trainloader):
            
            imgs, texts, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)

            img_embeddings = resnet50(imgs).to(device).squeeze(2).squeeze(2)
            text_embeddings = torch.stack([get_encoding(text, bert, device) for text in texts]).to(device)
            
            opt.zero_grad()
            outputs, imgs_f, texts_f = model(img_embeddings, text_embeddings)

            classification_loss = classification_loss_fn(outputs, labels)
            graph_loss = graph_loss_fn(outputs, labels, adj_matrix, device) 
            gap_loss = gap_loss_fn(imgs_f, texts_f)
            loss= (classification_loss * args.classification_weight) + (graph_loss * args.graph_weight) + (gap_loss * args.gap_weight)
            
            loss_agg += loss.item()
            classification_loss_agg += classification_loss.item()
            graph_loss_agg += graph_loss.item()
            gap_loss_agg += gap_loss.item()
            
            loss.backward()
            opt.step()
            
            if (batch_id + 1) % args.log_interval == 0:
                mesg = "\tEpoch {}:  [{}/{}]\tloss: {:.6f}\tclf_avg_loss: {:.6f}\tgraph_avg_loss: {:.6f}\tgap_av_loss: {:.6f}\tloss_avg: {:.6f}".format(
                            e + 1, args.batch_size * (batch_id + 1), len(trainset), loss.item(),
                            classification_loss_agg/(batch_id + 1), graph_loss_agg/(batch_id + 1), gap_loss_agg/(batch_id + 1), loss_agg/(batch_id + 1))
                print(mesg)

        # Checkpointing model after every epoch
        if args.checkpoint_model_dir is not None and os.path.exists(args.checkpoint_model_dir):
            model.eval().cpu()
            ckpt_model_path = os.path.join(args.checkpoint_model_dir, 'checkpoint.pth.tar')
            torch.save({'epoch': e + 1, 'network_state_dict': model.state_dict(),
                        'optimizer' : opt.state_dict()}, ckpt_model_path)
            model.to(device).train()

    #Saving the model after the training is complete
    if args.save_model_dir is not None and os.path.exists(args.save_model_dir):
        model.eval().cpu()
        save_model_filename = "epoch_" + str(args.epochs) + "_" + str(args.batch_size) + ".pth.tar"
        save_model_path = os.path.join(args.save_model_dir, save_model_filename)
        torch.save(model.state_dict(), save_model_path)
        print("\nDone, trained model saved at", save_model_path)

def evaluate(args):
  device = torch.device("cuda:0" if args.cuda else 'cpu')
  
  dataframe = pd.read_csv(args.csv_file_path)
  processed_dataframe = process_dataframe(dataframe)

  X = processed_dataframe.loc[:, ['image', 'processed_text']].values
  y = processed_dataframe['mapped_classes'].values
    
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
  valset = ImageTextDataset(X, y, transform)
  valloader = torch.utils.data.DataLoader(valset, shuffle = False, batch_size = 1)

  resnet50 = models.resnet50(pretrained = True).to(device)
  resnet50 = torch.nn.Sequential(*(list(resnet50.children())[:-1])).to(device).eval()
  bert = BertModel.from_pretrained('bert-base-uncased').to(device).eval()
  
  model = Towers().to(device).eval()
  model.load_state_dict(torch.load(args.model))
  
  correct = 0
  val_loss = 0
  pred_label_pairs = []
  
  with torch.no_grad():
    for data in valloader:
      
        img, text, label = data
        img = img.to(device)
        label = label.to(device)

        img_embeddings = resnet50(img).to(device).squeeze(2).squeeze(2)
        text_embeddings = torch.stack([get_encoding(text, bert, device) for text in text]).to(device)

        pred, img_f, text_f = model(img_embeddings, text_embeddings)
      
        classification_loss = classification_loss_fn(pred, label) 
        gap_loss = gap_loss_fn(img_f, text_f)

        loss = classification_loss + gap_loss
        val_loss += loss.item()

        pred = pred.data.max(1)[1]
        correct += pred.eq(label.data).sum().item()

        pred_label_pairs.append((label.item(), pred.item()))

    val_loss /= len(valloader.dataset)
    val_accuracy = 100.0 * correct / len(valloader.dataset)
    pd.DataFrame(pred_label_pairs, columns = ['label', 'predicted']).to_csv('predictions.csv', header = True, index = False)

    return 'Loss: ' + str(val_loss) + ', Accuracy: ' + str(val_accuracy)

def check_paths(args):

    try:
        if args.save_model_dir is not None and not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()