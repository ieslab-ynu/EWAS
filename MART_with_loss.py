import os
import argparse
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import datetime
from models import *
import numpy as np
import attack_generator as attack
from utils import Logger
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Adversarial Training for MART')
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--milestones', nargs='+', default=[60, 90], type=int,help='learning rate decay epoch')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet_ras",
                    help="decide which network to use,choose from resnet_ras,wrn_ras")
parser.add_argument('--beta',type=float,default=6.0,help='regularization parameter')
parser.add_argument('--num_workers', type=int, default=0, help="dataloader number of workers")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--out_dir',type=str,default='./MART_results',help='dir of output')
parser.add_argument('--resume', type=str, default='', help='whether to resume training, default: None')
parser.add_argument('--lam', type=float, default=0.01, help='lambda to control l1')

args = parser.parse_args()

# settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf',
              lam=1.
              ):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                mask_adv,out_adv  = model(x_adv, y)
                loss_ce = F.cross_entropy(out_adv, y)+lam*F.cross_entropy(mask_adv, y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    mask_out, out = model(x_natural, y)

    mask_adv,out_adv  = model(x_adv, y)

    adv_probs = F.softmax(out_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(out_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)


    mask_loss_adv = 0.

    adv_probs_mask = F.softmax(mask_adv, dim=1)
    tmp1 = torch.argsort(adv_probs_mask, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    mask_loss_adv +=  (F.cross_entropy(mask_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs_mask + 1e-12), new_y))

    loss_adv += lam * mask_loss_adv

    nat_probs = F.softmax(out, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    extra_loss_robust = 0.
    # for idx, output in enumerate(extra_outputs_nat):
    nat_probs_extra = F.softmax(mask_adv, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    extra_loss_robust += (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs_mask + 1e-12), nat_probs_extra), dim=1) * (1.0000001 - true_probs))
    loss_robust += lam * extra_loss_robust

    loss = loss_adv + float(beta) * loss_robust

    return loss


def train(model, train_loader, optimizer):
    starttime = datetime.datetime.now()
    loss_sum = 0
    bp_count = 0


    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.cuda(), target.cuda()



        # calculate MART adversarial training loss
        loss = mart_loss(model,data,target,optimizer,step_size=args.step_size,epsilon=args.epsilon, perturb_steps= args.num_steps,beta=args.beta,lam=args.lam)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    bp_count_avg = bp_count / len(train_loader.dataset)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds

    return time, loss_sum, bp_count_avg



def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.num_workers)
num_classes = 10



print('==> Load Model')

if args.net == "wrn_ras":
    model = Wide_ResNet_RAS(num_classes=num_classes).cuda()
    net = "wrn_ras"
if args.net == "resnet_ras":
    model = ResNet_RAS(num_classes=num_classes).cuda()
    net = "resnet_ras"




model = torch.nn.DataParallel(model)
print(net)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestones,gamma=0.1)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

start_epoch = 0
# Resume
title = 'MART train'
if args.resume:
    # resume directly point to resnet18-c10.pth.tar e.g., --resume='./out-dir/resnet18-c10.pth.tar'
    print ('==> MART Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.last_epoch = start_epoch
    logger_test = Logger(os.path.join(out_dir, 'log_{}_lam{}_results.txt'.format(net,args.lam)), title=title, resume=True)
else:
    print('==> MART train')
    logger_test = Logger(os.path.join(out_dir,  'log_{}_lam{}_results.txt'.format(net,args.lam)), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc'])


test_nat_acc = 0
fgsm_acc = 0
test_pgd20_acc = 0
cw_acc = 0
best_epoch = 0
best_robust = 0
for epoch in range(start_epoch, args.epochs):
    train_time, train_loss, bp_count_avg = train(model, train_loader, optimizer)
    if epoch == 0 or (epoch+1) % 10 == 0 or (epoch >= 60):
        ## Evalutions the same as DAT.
        loss, test_nat_acc = attack.eval_clean(model, test_loader)
        loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,
                                            loss_fn="cent", category="Madry", rand_init=True,lam = args.lam)
        loss, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031,
                                                  step_size=0.031 / 4, loss_fn="cent", category="Madry", rand_init=True,lam = args.lam)
        loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,
                                          loss_fn="cw", category="Madry", rand_init=True, num_classes=num_classes,lam = args.lam)

        print(
            'Epoch: [%d | %d] | lr: %.2f | Train Time: %.2f s | BP Average: %.2f | Natural Test Acc %.2f | FGSM Test Acc %.2f | PGD20 Test Acc %.2f | CW Test Acc %.2f |\n' % (
            epoch + 1,
            args.epochs,
            optimizer.param_groups[0]['lr'],
            train_time,
            bp_count_avg,
            test_nat_acc,
            fgsm_acc,
            test_pgd20_acc,
            cw_acc)
            )

        if test_pgd20_acc>=best_robust:
            best_robust = test_pgd20_acc
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'bp_avg': bp_count_avg,
                'test_nat_acc': test_nat_acc,
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer': optimizer.state_dict(),
            },
                filename='best_{}_lam{}.pth.tar'.format(net, args.lam))

        logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc])
        print('Best result @ {} , {} '.format(best_epoch+1, best_robust))
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'bp_avg': bp_count_avg,
        'test_nat_acc': test_nat_acc,
        'test_pgd20_acc': test_pgd20_acc,
        'optimizer': optimizer.state_dict(),
    },
        filename='last_{}_lam{}.pth.tar'.format(net, args.lam))

    scheduler.step()
