from sys import prefix
import os
import torch
import logging
import random
import numpy as np
import warnings
from args import get_parser
from dataset.data import data_prefetcher, AllData_DataFrame
import torch.distributed as dist
from models.sfcn_mini import SFCN
from models.dbnV2 import DBN
from models.resnetV2 import resnet18, resnet34, resnet50
from models.densenetV2 import densenet121, densenet201
from models.context import Refine as Refine
from apex import amp
import torch.nn.functional as F
from apex.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import reduce_mean,adjust_learning_rate, AverageMeter, ProgressMeter, my_KLDivLoss
root = "/data5/yang/brain"

def initialize():
    # get args
    args = get_parser()

    # warnings
    warnings.filterwarnings("ignore")

    # logger
    logger = logging.getLogger(__name__)

    # set seed
    seed = int(1111)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # initialize logger
    logger.setLevel(level = logging.INFO)

    if not os.path.exists(os.path.join(root,"logs")):
        os.makedirs(os.path.join(root,"logs"))

    handler = logging.FileHandler(os.path.join(root,"logs/%s.txt" % args.env_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return args, logger

def main():
    config, logger = initialize()
    config.nprocs = torch.cuda.device_count()
    main_worker(config, logger)

def weight_func(func,x,y,weight):
    n=x.shape[0]
    ret = 0.
    for i in range(n):
        ret += func(x[i],y[i]) * weight[i]
    return ret / n
def main_worker(config, logger):
    model_names = ["resnet18", "resnet50", "vgg", "dense121", "sfcn", "dbn"]
    models = [resnet18, resnet50,vgg16_bn, densenet121, SFCN, DBN]

    best_acc1 = -99.0
    best_acc2 = -99.0
    best_auc = -99.0
    dist.init_process_group(backend='nccl')
    # create model
    model_t = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)
    model_s = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)
    
    if config.arch == 'resnet18':
        refine = Refine([64,128,256,512],config)
    elif config.arch == 'resnet50':
        refine = Refine([256,512,1024,2048],config)
    elif config.arch == 'dense121':
        refine = Refine([128,256,512,1024],config)
    elif config.arch == 'dbn':
        refine = Refine([320,1088,2080,2080],config)
    
    torch.cuda.set_device(config.local_rank)

    # find the best teacher epoch
    dirs = f"checkpoints/{config.arch}_teacher_fold-{config.fold}-model-{config.arch}"
    files = os.listdir(os.path.join(root,dirs))
    trained_epoch = [int(f.split("_")[-2]) for f in files]
    max_epoch = max(trained_epoch)
    use_file = files[trained_epoch.index(max_epoch)]
    print(use_file)

    checkpoint = torch.load(os.path.join(root,dirs,use_file), map_location='cpu') 
    model_t.load_state_dict(checkpoint['state_dict'])

    train_list = torch.nn.ModuleList()
    train_list.append(model_s)
    train_list.append(refine)
    
    model_t.cuda().eval()
    model_s.cuda()
    refine.cuda()

    logger.info("Loading finished.")
    config.batch_size = int(config.batch_size / config.nprocs)
    
    optimizer = torch.optim.Adam(train_list.parameters(),lr = config.lr,weight_decay = 0.0005)

    train_list, optimizer = amp.initialize(train_list, optimizer, opt_level=config.opt_level)
    model_s = DistributedDataParallel(model_s)
    
    cudnn.benchmark = True

    # Data loading code
    train_data = AllData_DataFrame(os.path.join(root,"dataset/dataset.csv"),config, train = True)
    val_data = AllData_DataFrame(os.path.join(root,"dataset/dataset.csv"),config, train = False)
    #test_data = AllData_DataFrame("/data5/yang/brain/dataset/dataset.csv", train = False, test = True)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)

    train_loader = DataLoader(train_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = True, sampler = train_sampler)
    val_loader = DataLoader(val_data,config.batch_size,
                        shuffle=False,num_workers=8,pin_memory = False, sampler = val_sampler)
    #test_loader = DataLoader(test_data,config.batch_size,
    #                    shuffle=False,num_workers=8,pin_memory = False, sampler = test_sampler)

    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, config)
        import time
        # train for one epoch
        st_time = time.time()
        train(train_loader, model_t, model_s,refine, optimizer, epoch, config, logger)
        end_time = time.time()
        run_time = end_time - st_time
        logger.info(f"Running time for epoch: {run_time}")
        acc,auc = validate(val_loader,model_s, config, logger, prefix = "val")
        #test_acc = validate(test_loader,model_t, config, logger, prefix = "test")

        is_best = (acc > best_acc1) or (acc == best_acc1 and auc > best_auc)

        best_acc1 = max(acc, best_acc1)
        best_auc = max(auc, best_auc)
        #best_acc2 = max(test_acc, best_acc2)
        check_path = os.path.join(root,"checkpoints/mine/%s" % config.env_name)
        if not os.path.exists(check_path):
            try:
                os.makedirs(check_path)
            except:
                pass # multiple processors bug

        if is_best and config.local_rank == 0 and acc > 0.7:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model_s.module.state_dict(),
                    'best_acc1': best_acc1,
                    'amp': amp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            torch.save(state, os.path.join(root, f'{check_path}/{config.env_name}_epoch_{epoch}_{acc}'))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(train_loader, model_t,model_s,refine, optimizer, epoch, config,logger):
    losses = AverageMeter('Loss', ':.4e')
    loss_acc = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(len(train_loader), [losses, loss_acc],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)
    model_t.eval()
    model_s.train()

    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    logger.info(f"scale:{config.scale},tem:{config.T}")
    while images is not None:
        out_s = model_s(images, config.T)
        out_t = model_t(images, config.T)
        
        weight = abs(out_t['p'].max(1)[0] - target) / config.scale
        weight[weight >1] = 1
        weight = 1 - weight
        
        loss_refine = config.T * config.T * refine(out_t,out_s, weight)
        loss_kd = config.T * config.T * weight_func(torch.nn.KLDivLoss(),out_s['y_tem'], out_t['p'],weight)
        loss = F.nll_loss(out_s['y'], target) + loss_kd * config.alpha + loss_refine * config.beta

        acc = accuracy(out_s['y'],target)
        torch.distributed.barrier()
        reduced_loss = reduce_mean(loss, config.nprocs)
        reduced_acc = reduce_mean(acc, config.nprocs)
        
        losses.update(reduced_loss.item(), images.size(0))
        loss_acc.update(reduced_acc.item(), images.size(0))

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if i % config.print_freq == 0 and config.local_rank == 0:
            progress.display(i)

        i += 1

        images, target, yy, bc,indices = prefetcher.next()

    if config.local_rank == 0:
        logger.info(f"[train loss]: {round(float(losses.avg),4)}, [train acc]: {round(float(loss_acc.avg),4)}")


from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix
def validate(val_loader, model, config, logger, prefix = ""):

    loss_metric = AverageMeter('acc', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_metric], prefix='Test: ', logger = logger)
    model.eval()

    preds = []
    lbls = []
    lbl_onehot = np.zeros((len(lbls), len(np.unique(lbls))))
    for i in range(len(lbls)):
        lbl_onehot[i,lbls[i]]=1
    with torch.no_grad():
        prefetcher = data_prefetcher(val_loader)
        images, target, yy, bc,indices = prefetcher.next()
        while images is not None:
            out = model(images)
            acc = accuracy(out['y'], target)
            preds.extend(out['y'].max(1)[1].cpu().detach().numpy())
            lbls.extend(target.cpu().detach().numpy())

            torch.distributed.barrier()
            reduced_acc = reduce_mean(acc, config.nprocs)
            loss_metric.update(reduced_acc.item(), images.size(0))

            images, target, yy, bc, indices = prefetcher.next()
    auc = round(roc_auc_score(lbls,preds),4)
    cm = confusion_matrix(lbls,preds)
    eval_sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
    eval_spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)
    if config.local_rank == 0:
        logger.info(f"\033[32m >>>>>>>> [{prefix}-acc]: {round(float(loss_metric.avg),4)}\
                    | auc: {auc} | recall: {eval_sen} | spe: {eval_spe} \033[0m")

    return round(loss_metric.avg,4),auc


if __name__ == '__main__':
    main()
