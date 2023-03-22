import warnings
from threading import Thread

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr, emojis
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel

import  os

LOGGERS = ('csv', 'tb', 'wandb')  # text-file, TensorBoard, Weights & Biases

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                     'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',  # metrics
                     'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                     'x/lr0', 'x/lr1', 'x/lr2']  # params
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize ssmokenet runs (RECOMMENDED)"
            print(emojis(s))

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get('wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob('*labels*.jpg')  # training labels
        if self.wandb:
            self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})

    # def on_train_batch_end(self, ni, model, imgs, targets, paths, plots):
    def on_train_batch_end(self, epoch_i, batch_i, imgs, targets):
        # Callback runs on train batch end
        paths=os.getcwd()
        f = self. args.output_dir/ f'train{epoch_i}_batch{batch_i}.jpg'  # filename
        a=targets.shape[0]
        b=targets.shape[1]
        c=torch.zeros(a,b,1)  # 12,120,1
        for i in range(len(targets)):
            c[i,:,:]=i
        targets_news=torch.cat([c,targets],dim=2)  # 12,120,6
        targets_news=targets_news.reshape(a*b,-1)  # 1440,6
        indices=targets_news[:,4]>0  # 框宽>0
        targets_news=targets_news[indices]

        # Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
        Thread(target=plot_images, args=(imgs, targets_news, None, f), daemon=True).start()  # paths=None


    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        # Callback runs on val end
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, mloss, results, lr, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        vals = list(mloss) + list(results) + lr
        x = {k: v for k, v in zip(self.keys, vals)}  # dict
        if self.csv:
            file = self.save_dir / 'results.csv'  # 写result.csv文件
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch):
        # Callback runs on training end
        if plots:
            plot_results(dir=self.save_dir)  # save results.png
        files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
        files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()]  # filter

        if self.tb:
            from PIL import Image
            import numpy as np
            for f in files:
                self.tb.add_image(f.stem, np.asarray(Image.open(f)), epoch, dataformats='HWC')

        if self.wandb:
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            wandb.log_artifact(str(best if best.exists() else last), type='model',
                               name='run_' + self.wandb.wandb_run.id + '_model',
                               aliases=['latest', 'best', 'stripped'])
            self.wandb.finish_run()
