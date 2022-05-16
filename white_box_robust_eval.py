import os
import argparse
import torchvision
from torchvision import transforms
from models import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import attack_generator as attack


parser = argparse.ArgumentParser(description='PyTorch Feature Visualization')

parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--num_workers', type=int, default=0, help="dataloader number of workers")
parser.add_argument('--net', type=str, default="resnet_ewas",
                    help="decide which network to use,choose from resnet_ewas,wrn_ewas")
parser.add_argument('--ckps_path', type=str, default=None, help='load checkpoint path')
parser.add_argument('--lam', type=float, default=0.01, help='lambda to control l1')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

print('==> Load Test Data')
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=args.num_workers)
num_classes = 10



print('==> Load Model')

if args.net == "wrn_ewas":
    model = Wide_ResNet_RAS(num_classes=num_classes).cuda()
    net = "wrn_ewas"
if args.net == "resnet_ewas":
    model = ResNet_RAS(num_classes=num_classes).cuda()
    net = "resnet_ewas"


model = torch.nn.DataParallel(model)
print(net)


print(args.ckps_path)
assert os.path.isfile(args.ckps_path)
out_dir = os.path.dirname(args.ckps_path)
checkpoint = torch.load(args.ckps_path)
model.load_state_dict(checkpoint['state_dict'])

print('start evaluating!')
loss, test_nat_acc = attack.eval_clean(model, test_loader)
loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,
                                    loss_fn="cent", category="Madry", rand_init=True,lam = args.lam)
loss, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031,
                                          step_size=0.031 / 4, loss_fn="cent", category="Madry", rand_init=True,lam = args.lam)
loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,
                                  loss_fn="cw", category="Madry", rand_init=True, num_classes=num_classes,lam = args.lam)
print(
    'Natural Test Acc %.4f | FGSM Test Acc %.4f | PGD20 Test Acc %.4f | CW Test Acc %.4f |\n' % (
        test_nat_acc,
        fgsm_acc,
        test_pgd20_acc,
        cw_acc)
)
