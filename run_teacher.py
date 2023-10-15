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
from models.vgg import vgg16_bn
from models.dbn import DBN
from models.resnet import resnet18, resnet34, resnet50
from models.densenet import densenet121, densenet201
from apex import amp
import torch.nn.functional as F
from apex.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.utils import reduce_mean,adjust_learning_rate, AverageMeter, ProgressMeter, my_KLDivLoss
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # initialize logger
    logger.setLevel(level = logging.INFO)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    handler = logging.FileHandler("logs/%s.txt" % args.env_name)
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

def main_worker(config, logger):
    model_names = ["resnet18", "resnet50", "vgg", "dense121", "sfcn", "dbn"]
    models = [resnet18, resnet50,vgg16_bn, densenet121, SFCN, DBN]

    best_acc1 = -99.0
    best_acc2 = -99.0
    best_auc = -99.0
    dist.init_process_group(backend='nccl')
    # create model
    model_t = models[model_names.index(config.arch)](output_dim=2, mode = config.mode)

    torch.cuda.set_device(config.local_rank)
    model_t.cuda().eval()


    config.batch_size = int(config.batch_size / config.nprocs)
    
    optimizer = torch.optim.Adam(model_t.parameters(),lr = config.lr,weight_decay = 0.0001)
    #optimizer.load_state_dict(checkpoint['optimizer'])    

    model_t, optimizer = amp.initialize(model_t, optimizer, opt_level=config.opt_level)
    #amp.load_state_dict(checkpoint['amp'])
    model_t = DistributedDataParallel(model_t)
    
    cudnn.benchmark = True

    # Data loading code
    if config.samples == 200:
        data_path = f"/data5/yang/brain/dataset/dataset_{config.fold}_200.csv"
    elif config.samples == 400:
        data_path = f"/data5/yang/brain/dataset/dataset_{config.fold}_400.csv"
    elif config.samples == 600:
        data_path = f"/data5/yang/brain/dataset/dataset_{config.fold}_600.csv"
    elif config.samples == 800:
        data_path = f"/data5/yang/brain/dataset/dataset_{config.fold}_800.csv"
    else:
        data_path = f"/data5/yang/brain/dataset/dataset_{config.fold}_all.csv"

    train_data = AllData_DataFrame(data_path,config, train = True)
    val_data = AllData_DataFrame(data_path,config, train = False)
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

        # train for one epoch
        train(train_loader, model_t, optimizer, epoch, config, logger)
            
        acc,auc = validate(val_loader,model_t, config, logger, prefix = "val")
        #test_acc = validate(test_loader,model_t, config, logger, prefix = "test")

        is_best = (acc > best_acc1) or (acc == best_acc1 and auc > best_auc)

        best_acc1 = max(acc, best_acc1)
        best_auc = max(auc, best_auc)
        #best_acc2 = max(test_acc, best_acc2)
        if not os.path.exists("./checkpoints/%s" % config.env_name):
            try:
                os.makedirs("./checkpoints/%s" % config.env_name)
            except:
                pass # multiple processors bug

        if is_best and config.local_rank == 0:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model_t.module.state_dict(),
                    'best_acc1': best_acc1,
                    'amp': amp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            torch.save(state, './checkpoints/%s/%s_epoch_%s_%s' % (config.env_name, config.env_name, epoch, acc))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(train_loader, model_t, optimizer, epoch, config,logger):
    #return
    losses = AverageMeter('Loss', ':.4e')
    loss_acc = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(len(train_loader), [losses, loss_acc],
                             prefix="Epoch: [{}]".format(epoch), logger = logger)

    model_t.train()

    prefetcher = data_prefetcher(train_loader)
    images, target, yy, bc, indices = prefetcher.next()
    i = 0
    optimizer.zero_grad()
    optimizer.step()
    while images is not None:
        out_t = model_t(images)

        loss = F.nll_loss(out_t, target)
        acc = accuracy(out_t,target)
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
            acc = accuracy(out, target)
            #loss_auc = roc_auc_score()
            preds.extend(out.max(1)[1].cpu().detach().numpy())
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
