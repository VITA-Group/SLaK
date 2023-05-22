# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.nn.functional as F
import utils
import torch.nn as nn

class NKDLoss(nn.Module):

    """PyTorch version of NKD: `Rethinking Knowledge Distillation via Cross-Entropy` """

    def __init__(self,
                 temp=1.0,
                 alpha=1.,
                 ):
        super(NKDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha

    def forward(self, logit_s, logit_t, gt_label):
        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        y_i = F.softmax(logit_s, dim=1)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        y_t = torch.gather(y_i, 1, label)
        w_t = torch.gather(t_i, 1, label).detach()

        mask = torch.zeros_like(logit_s).scatter_(1, label, 1).bool()
        logit_s = logit_s - 1000 * mask
        logit_t = logit_t - 1000 * mask
        
        # N*class
        T_i = F.softmax(logit_t/self.temp, dim=1)
        S_i = F.softmax(logit_s/self.temp, dim=1)
        # N*1
        T_t = torch.gather(T_i, 1, label)
        S_t = torch.gather(S_i, 1, label)
        # N*class 
        np_t = T_i/(1-T_t)
        np_s = S_i/(1-S_t)
        np_t[T_i==T_t] = 0
        np_s[T_i==T_t] = 1

        soft_loss = - (w_t * torch.log(y_t)).mean() 
        distributed_loss =  (np_t * torch.log(np_s)).sum(dim=1).mean()
        distributed_loss = - self.alpha * (self.temp**2) * distributed_loss

        return soft_loss + distributed_loss 

def loss_kd(preds, labels, teacher_preds,T,hard=False,alpha=0.1):
    #T = 1
    if hard:
        y_t=teacher_preds.max(1)[-1]
        loss=F.cross_entropy(preds, labels) * 0.5+F.cross_entropy(preds, y_t) * 0.5
    else:
        #alpha = 0.1
        loss = F.kl_div(F.log_softmax(preds / T, dim=1), F.softmax(teacher_preds / T, dim=1),
                        reduction='batchmean') * T * T * alpha + F.cross_entropy(preds, labels) * (1. - alpha)
    return loss

swin_kernel_dict={0:7,1:7,2:14,3:28}
slak_kernel_dict={0:7,1:14,2:28,3:56}
vit_kernel_dict={0:14}
vit_dict={3:768}
convnext_kernel_dict={0:7,1:14,2:28,3:56}
swin_dict={0:192,1:384,2:768,3:768}
convnext_dict={0:96,1:192,2:384,3:768}
resnet_dict={0:256,1:512,2:1024,3:2048}
slak_dict={0:124,1:249,2:499,3:998}

def train_one_epoch(model: torch.nn.Module,model_convxt: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, mask=None,T=1,hard=False,args=None,MGDloss=None):
    model.eval()
    nkdloss=NKDLoss()
    model_convxt.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    if args.FDLoss_type=='smoothL1':
        loss_feat = nn.SmoothL1Loss(beta=2.0)
    elif args.FDLoss_type=='MSE':
        loss_feat = nn.MSELoss()
    else:
        assert False
    optimizer.zero_grad()
    Lnorms=[]
    if args.target_Lnorm:
        for i in range(args.feature_n):
            j=3-i
            if args.model=='swin':
                Lnorms.append(nn.LayerNorm((swin_dict[j],swin_kernel_dict[i],swin_kernel_dict[i]), elementwise_affine=False))
            elif args.model=='convnext':
                Lnorms.append(nn.LayerNorm((convnext_dict[j],convnext_kernel_dict[i],convnext_kernel_dict[i]), elementwise_affine=False))
            elif args.model=='SLaK_tiny':
                Lnorms.append(nn.LayerNorm((slak_dict[j],slak_kernel_dict[i],slak_kernel_dict[i]), elementwise_affine=False))
            elif args.model=='vit':
                Lnorms.append(nn.LayerNorm((vit_dict[j],vit_kernel_dict[i],vit_kernel_dict[i]), elementwise_affine=False))
            else:
                assert False

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq

        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                
                if args.distill_type!="None":
                    output_t = model(samples)
                output=model_convxt(samples)
                #loss = criterion(output, targets)
                if 'FD' in args.distill_type :
                    loss=criterion(output, targets)
                    if args.model=='swin':
                        target=model.module.feature.detach()
                        target=target.reshape(target.size(0),7,7, target.size(2))
                        target=target.transpose(2, 3).transpose(1, 2)
                        loss_dis=loss_feat(model_convxt.module.project_swin(),target)
                    elif args.model=='vit' or args.model=='vitdeit':
                        target=model.module.feature.detach()
                        target=target.reshape(target.size(0),14,14, target.size(2))
                        target=target.transpose(2, 3).transpose(1, 2)
                        loss_dis=loss_feat(model_convxt.module.project_vit(),target)
                    elif args.model=='SLaK_tiny':
                        target=model.module.feature.detach()
                        loss_dis=loss_feat(model_convxt.module.project_slak(),target)
                    else:
                        loss_dis=loss_feat(model_convxt.module.feature,model.module.feature.detach())

                    if 'NKD' in args.distill_type:
                        loss=nkdloss(output,output_t,targets)+loss_dis*args.lr_fd
                    else:    
                        if 'KD' in args.distill_type:
                            loss=loss_kd(output,targets,output_t,T,hard=hard,alpha=args.alpha)+loss_dis*args.lr_fd
                        else:   
                            loss=loss+loss_dis*args.lr_fd                
                elif args.distill_type=='KD':
                    loss=loss_kd(output,targets,output_t,T,hard=hard,alpha=args.alpha)
                elif args.distill_type=='NKD':
                    loss=nkdloss(output,output_t,targets)
                elif args.distill_type=='None':
                    loss=criterion(output, targets)
                else:
                    assert False
        else: # full precision
            if args.distill_type!="None":
                output_t = model(samples)
            output=model_convxt(samples)
          
            
            if args.distill_type=='KD':
                loss=loss_kd(output,targets,output_t,T,hard=hard,alpha=args.alpha)
            elif args.distill_type=='NKD':
                loss=nkdloss(output,output_t,targets)
            elif args.distill_type=='None':
                    loss=criterion(output,targets)     
            else:
                assert False

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model_convxt.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model_convxt, mask)

        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if mask:
                    mask.step()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model_convxt, mask)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
