import pdb
import sys, time, os, argparse, warnings, glob, torch
from tools import *
from model import *
from dataLoader import *

# Training settings
parser = argparse.ArgumentParser(description = "Stage I, self-supervsied speaker recognition with contrastive learning.")
parser.add_argument('--max_frames',        type=int,   default=180,          help='Input length to the network, 1.8s')
parser.add_argument('--batch_size',        type=int,   default=300,          help='Batch size, bigger is better')
parser.add_argument('--num_frames', type=int, default=300, help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--sample_rate', type=int,  default=32000,   help='Set your sample_rate')
parser.add_argument('--n_cpu',             type=int,   default=4,            help='Number of loader threads')
parser.add_argument('--test_interval',     type=int,   default=1,            help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',         type=int,   default=80,           help='Maximum number of epochs')
parser.add_argument('--lr',                type=float, default=0.001,        help='Learning rate')
parser.add_argument("--lr_decay",          type=float, default=0.95,         help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--initial_model',     type=str,   default="exp/exp4/model/model000000041.model",           help='Initial model path')
parser.add_argument('--save_path',         type=str,   default="",           help='Path for model and scores.txt')
parser.add_argument('--train_list',        type=str,   default="",           help='Path for Vox2 list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--test_list',          type=str,   default="",           help='Path for Vox_O list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--train_path',        type=str,   default="",           help='Path to the Vox2 set')
parser.add_argument('--test_path',          type=str,   default="",           help='Path to the Vox_O set')
parser.add_argument('--musan_path',        type=str,   default="",           help='Path to the musan set')
parser.add_argument('--eval',              dest='eval', action='store_true', help='Do evaluation only')
parser.add_argument('--label_num',         type=int,   default=4,            help='Number of label')
args = parser.parse_args()

# Initialization
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
os.makedirs(model_save_path, exist_ok = True)
scorefile = open(args.save_path+"/scores.txt", "a+")
it = 1

Trainer = model(**vars(args)) # Define the framework
modelfiles = glob.glob('%s/model0*.model'%model_save_path) # Search the existed model files
modelfiles.sort()

if(args.initial_model != ""): # If initial_model is exist, system will train from the initial_model
    Trainer.load_network(args.initial_model)
elif len(modelfiles) >= 1: # Otherwise, system will try to start from the saved model&epoch
    Trainer.load_network(modelfiles[-1])
    print(modelfiles[-1])
    exit()
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

if 'new' in args.test_list.split('/')[-1]:
    import dataLoader_v2
    test_dataset = dataLoader_v2.get_testloader(args)
else:
    test_dataset = get_testloader(args)

if args.eval == True: # Do evaluation only
    # EER, minDCF = Trainer.evaluate_network(**vars(args))
    # print('EER %2.4f, minDCF %.3f\n'%(EER, minDCF))
    avg_closs, Acc = Trainer.evaluate_network(loader=test_dataset)
    print('avg_closs %2.4f, Acc %2.3f\n' % (avg_closs, Acc))
    quit()

if 'new' in args.train_list.split('/')[-1]:
    import dataLoader_v2
    trainLoader = dataLoader_v2.get_loader(args)
else:
    trainLoader = get_loader(args) # Define the dataloader

while it < args.max_epoch:
    # Train for one epoch
    loss, traineer, lr = Trainer.train_network(loader=trainLoader, epoch = it)

    # Evaluation every [test_interval] epochs, record the training loss, training acc, evaluation EER/minDCF
    if it % args.test_interval == 0:
        Trainer.save_network(model_save_path+"/model%09d.model"%it)
        avg_closs, Acc = Trainer.evaluate_network(loader=test_dataset)
        # print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, Acc %2.2f, LOSS %f, EER %2.4f, minDCF %.3f"%( lr, traineer, loss, EER, minDCF))
        # scorefile.write("Epoch %d, LR %f, Acc %2.2f, LOSS %f, EER %2.4f, minDCF %.3f\n"%(it, lr, traineer, loss, EER, minDCF))
        print(time.strftime("%Y-%m-%d %H:%M:%S"),"LR %f, Acc %2.2f, LOSS %f, avg_closs %2.4f, Acc %2.3f" % (lr, traineer, loss, avg_closs, Acc))
        scorefile.write("Epoch %d, LR %f, Acc %2.2f, LOSS %f, avg_closs %2.4f, Acc %2.3f\n"%(it, lr, traineer, loss, avg_closs, Acc))
        scorefile.flush()
    # Otherwise, recored the training loss and acc
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, Acc %2.2f, LOSS %f"%( lr, traineer, loss))
        scorefile.write("Epoch %d, LR %f, Acc %2.2f, LOSS %f\n"%(it, lr, traineer, loss))
        scorefile.flush()

    it += 1
    print("")