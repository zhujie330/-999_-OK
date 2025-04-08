import argparse
import os
import utils
import ast

class TrainOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        # data augmentation
        parser.add_argument('--task',default='train', type=str)
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0.5)
        parser.add_argument('--blur_sig', default='0.0,3.0')
        parser.add_argument('--jpg_prob', type=float, default=0.5)
        parser.add_argument('--jpg_method', default='cv2,pil')
        parser.add_argument('--jpg_qual', default='30,100')
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--CropSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--detect_method', type=str,default='CNNSpot', help='choose the detection method')
        parser.add_argument('--dataroot', default='/home/zhangruixuan/dataset/FF++', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        #parser.add_argument('--classes', default='airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse', help='image classes to train on')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='loss weight for l2 reg')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
        parser.add_argument('--lr-min', default=0., type=float)
        parser.add_argument('--lr-max', default=0.2, type=float)
        parser.add_argument('--weight-decay', default=5e-4, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--epsilon', default=8, type=int)
        parser.add_argument('--alpha', default=2/255, type=float, help='FGSM Step size')
        parser.add_argument('--pgd_alpha',default=2/255, type=float, help='PGD Step size')
        parser.add_argument('--pgd_step',default=10, type=int, help='PGD Step')
        parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
                            help='Perturbation initialization method')
        parser.add_argument('--out_dir', default='outdir', type=str, help='Output directory')
        parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
        parser.add_argument('--adv_train', action='store_true', help='adverarial training or normal training')
        parser.add_argument('--adv_attack',default='fgsm', type=str, help='adversarial')
        parser.add_argument('--loss_freq', default=1, type=int, help='loss frequency')
        return parser
    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.results_dir, opt.name)
        utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = True   # train or test
        opt.isVal = False
        #opt.classes = opt.classes.split(',')

        # result dir, save results and opt
        opt.results_dir=f"./results/{opt.detect_method}"
        utils.mkdir(opt.results_dir)



        if print_options:
            self.print_options(opt)



        # additional

        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt

class TestOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):


        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_sig', default='1.0')
        parser.add_argument('--jpg_method', default='pil')
        parser.add_argument('--jpg_qual', default='95')

        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--CropSize', type=int, default=224, help='scale images to this size')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        parser.add_argument('--model_path',type=str,default='./weights/classifier/CNNSpot.pth',help='the path of detection model')
        # parser.add_argument('--is_single',action='store_true',help='evaluate image by image')
        parser.add_argument('--detect_method', type=str,default='CNNSpot', help='choose the detection method')
        parser.add_argument('--noise_type', type=str,default=None, help='such as jpg, blur and resize')

        # path of processing model
        parser.add_argument('--LNP_modelpath',type=str,default='./weights/preprocessing/sidd_rgb.pth',help='the path of LNP pre-trained model')
        parser.add_argument('--DIRE_modelpath',type=str,default='./weights/preprocessing/lsun_bedroom.pt',help='the path of DIRE pre-trained model')
        parser.add_argument('--LGrad_modelpath', type=str,default='./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth', help='the path of LGrad pre-trained model')

        self.initialized = True

        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        file_name = os.path.join(opt.results_dir, f'{opt.noise_type}opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = False   # train or test
        opt.isVal = False

        # result dir, save results and opt
        opt.results_dir=f"./results/{opt.detect_method}"
        utils.mkdir(opt.results_dir)



        if print_options:
            self.print_options(opt)



        # additional

        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
