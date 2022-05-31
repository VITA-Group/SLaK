from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from funcs import redistribution_funcs, growth_funcs, prune_funcs

def add_sparse_args(parser):
    # hyperparameters for Sparse Training
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: False.')
    parser.add_argument('--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, random_unfired, and momentum_neuron.')
    parser.add_argument('--prune', type=str, default='magnitude', help='Prune mode / pruning mode. Choose from: magnitude, SET.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--prune-rate', type=float, default=0.50, help='The pruning rate / prune rate.')
    parser.add_argument('--density', type=float, default=0.5, help='The density of the overall sparse network.')
    parser.add_argument('--verbose', action='store_true', help='Prints verbose status of pruning/growth algorithms.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse-init', type=str, default='ERK', help='sparse initialization')
    parser.add_argument('--multiplier', type=float, default=1.0, metavar='N', help='extend training time by multiplier times')

    # hyperparameters for GraNet
    parser.add_argument('-u', '--update-frequency', type=int, default=2500, metavar='N', help='how many iterations to train between mask update')
    parser.add_argument('--init-density', type=float, default=0.5, help='The initial density of sparse networks')
    parser.add_argument('--final-density', type=float, default=0.20, help='The target density of sparse networks.')
    parser.add_argument('-i', '--init-prune-epoch', type=int, default=0, help='The starting epoch of gradual pruning.')
    parser.add_argument('-f', '--final-prune-epoch', type=int, default=100, help='The ending epoch of gradual pruning.')
    parser.add_argument('--method', type=str, default='DST', help='method name: DST, GraNet, GMP')
    parser.add_argument('--rm-first', action='store_true', help='Keep the first layer dense.')
    parser.add_argument('--only-L', action='store_true', help='only sparsify large kernels.')

class CosineDecay(object):
    """Decays a pruning rate according to a cosine schedule

    This class is just a wrapper around PyTorch's CosineAnnealingLR.
    """
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1, init_step=0):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)
        if init_step!=0:
            for i in range(init_step):
                self.cosine_stepper.step()
    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, prune_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    """Anneals the pruning rate linearly with each step."""
    def __init__(self, prune_rate, T_max):
        self.steps = 0
        self.decrement = prune_rate/float(T_max)
        self.current_prune_rate = prune_rate

    def step(self):
        self.steps += 1
        self.current_prune_rate -= self.decrement

    def get_dr(self, prune_rate):
        return self.current_prune_rate

def prefetched_loader(loader, fp16):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
    if fp16:
        mean = mean.half()
        std = std.half()

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            if fp16:
                next_input = next_input.half()
            else:
                next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target

class Masking(object):
    """Wraps PyTorch model parameters with a sparse mask.

    Creates a mask for each parameter tensor contained in the model. When
    `apply_mask()` is called, it applies the sparsity pattern to the parameters.

    Basic usage:
        optimizer = torchoptim.SGD(model.parameters(),lr=args.lr)
        decay = CosineDecay(args.prune_rate, len(train_loader)*(args.epochs))
        mask = Masking(optimizer, prune_rate_decay=decay)
        model = MyModel()
        mask.add_module(model)

    Removing layers: Layers can be removed individually, by type, or by partial
    match of their name.
      - `mask.remove_weight(name)` requires an exact name of
    a parameter.
      - `mask.remove_weight_partial_name(partial_name=name)` removes all
        parameters that contain the partial name. For example 'conv' would remove all
        layers with 'conv' in their name.
      - `mask.remove_type(type)` removes all layers of a certain type. For example,
        mask.remove_type(torch.nn.BatchNorm2d) removes all 2D batch norm layers.
    """
    def __init__(self, optimizer,train_loader, prune_rate_decay, prune_rate=0.5, prune_mode='magnitude', growth_mode='momentum', redistribution_mode='momentum', verbose=False, fp16=False, args=False):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))
        self.args = args
        self.device = torch.device(args.device)
        self.growth_mode = growth_mode
        self.prune_mode = prune_mode
        self.redistribution_mode = redistribution_mode
        self.prune_rate_decay = prune_rate_decay
        self.verbose = verbose
        self.train_loader = train_loader
        self.growth_func = growth_mode
        self.prune_func = prune_mode
        self.redistribution_func = redistribution_mode

        self.global_growth = False
        self.global_prune = False

        self.masks = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.name2removed = {}

        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.prune_rate = prune_rate
        self.name2prune_rate = {}
        self.steps = 0
        self.start_name = None


        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = self.args.update_frequency
        self.half = fp16
        self.name_to_32bit = {}


    def init_optimizer(self):
        if 'fp32_from_fp16' in self.optimizer.state_dict():
            for (name, tensor), tensor2 in zip(self.modules[0].named_parameters(), self.optimizer.state_dict()['fp32_from_fp16'][0]):
                self.name_to_32bit[name] = tensor2
            self.half = True

    def init(self, mode='ERK', density=0.05, erk_power_scale=1.0):
        self.init_growth_prune_and_redist()
        self.init_optimizer()
        self.density = density
        if self.args.method == 'GMP':
            print('initialized with GMP, ones')
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).to(self.device)
                    self.baseline_nonzero += (self.masks[name] != 0).sum().int().item()

        elif mode == 'uniform':
            print('initialized with uniform')
            # initializes each layer with a constant percentage of dense weights
            # each layer will have weight.numel()*density weights.
            # weight.numel()*density == weight.numel()*(1.0-sparsity)
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name][:] = (torch.rand(weight.shape) < density).float().data.to(self.device)
                    self.baseline_nonzero += weight.numel()*density

        elif mode == 'resume':
            print('initialized with resume')
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item())
                    if name in self.name_to_32bit:
                        print('W2')
                    self.masks[name][:] = (weight != 0.0).float().data.to(self.device)
                    self.baseline_nonzero += weight.numel()*density
        elif mode == 'ERK_plus':
            print('initialize by ERK_plus')
            total_params = 0
            self.baseline_nonzero = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
                self.baseline_nonzero += weight.numel() * density
            for name in self.masks.copy():
                if 'head.weight' in name:
                    total_params = total_params - self.masks[name].numel()
                    density = (self.baseline_nonzero - self.masks[name].numel()) / total_params
                    self.masks.pop(name)

            is_epsilon_valid = False
            dense_layers = set()
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}

                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density
                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        if len(mask.shape) != 2:
                            raw_probabilities[name] = (
                                                              np.sum(mask.shape) / np.prod(mask.shape)
                                                      ) ** erk_power_scale
                        else:
                            raw_probabilities[name] = (
                                                              np.sum(mask.shape) / np.prod(mask.shape)
                                                      ) ** erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True
            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()
                total_nonzero += density_dict[name] * mask.numel()

            # for name, weight in self.module.named_parameters():
            #     if 'head.weight' in name:
            #         self.masks[name] = (torch.rand(weight.shape) < self.fc_density).float().data.cuda()
            #         total_nonzero += self.fc_density * weight.numel()
            #         total_params += weight.numel()
            #         print(
            #             f"layer: {name}, shape: {self.masks[name].shape}, density: {self.fc_density}"
            #         )
            print(f"Overall sparsity {total_nonzero / total_params}")
        elif mode == 'ERK':
            print('initialize by fixed_ERK')
            total_params = 0
            self.baseline_nonzero = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()
                self.baseline_nonzero += weight.numel() * density
            is_epsilon_valid = False
            # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.to(self.device)

                total_nonzero += density_dict[name] * mask.numel()
            print(f"Overall sparsity {total_nonzero / total_params}")

        total_size = 0
        sparse_size = 0
        dense_layers = []
        for name, weight in self.masks.items():
            dense_weight_num = weight.numel()
            sparse_weight_num = (weight != 0).sum().int().item()
            total_size += dense_weight_num
            sparse_size += sparse_weight_num
            layer_density = sparse_weight_num / dense_weight_num
            if layer_density >= 0.99: dense_layers.append(name)
            print(f'Density of layer {name} with tensor {weight.size()} is {layer_density}')
        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))

        # masks of layers with density=1 are removed
        for name in dense_layers:
            self.masks.pop(name)
            print(f"pop out layer {name}")

        self.apply_mask()

    def init_growth_prune_and_redist(self):
        if isinstance(self.growth_func, str) and self.growth_func in growth_funcs:
            if 'global' in self.growth_func: self.global_growth = True
            self.growth_func = growth_funcs[self.growth_func]
        elif isinstance(self.growth_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Growth mode function not known: {0}.'.format(self.growth_func))
            print('Use either a custom growth function or one of the pre-defined functions:')
            for key in growth_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown growth mode.')

        if isinstance(self.prune_func, str) and self.prune_func in prune_funcs:
            if 'global' in self.prune_func: self.global_prune = True
            self.prune_func = prune_funcs[self.prune_func]
        elif isinstance(self.prune_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Prune mode function not known: {0}.'.format(self.prune_func))
            print('Use either a custom prune function or one of the pre-defined functions:')
            for key in prune_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown prune mode.')

        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')


    def step(self):
        self.optimizer.step()
        self.apply_mask()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr(self.prune_rate)
        self.steps += 1

        if self.prune_every_k_steps is not None:
            if self.args.method == 'GraNet':
                if self.steps >= (self.args.init_prune_epoch * self.train_loader) and \
                        self.steps % self.prune_every_k_steps == 0:
                    print('*************************************Pruning*************************************')
                    self.pruning(self.steps)
                    print('*********************************Dynamic Sparsity********************************')
                    self.truncate_weights()
                    self.print_nonzero_counts()
            elif self.args.method == 'GraNet_uniform':
                if self.steps >= (self.args.init_prune_epoch * int(len(self.train_loader)/self.args.world_size) * self.args.multiplier) and \
                        self.steps % self.prune_every_k_steps == 0:
                    print('*************************************Pruning*************************************')
                    self.pruning_uniform(self.steps)
                    print('*********************************Dynamic Sparsity********************************')
                    self.truncate_weights()
                    self.print_nonzero_counts()
            elif self.args.method == 'DST':
                if self.steps % self.prune_every_k_steps == 0:
                    print('*********************************Dynamic Sparsity********************************')
                    self.truncate_weights()
                    self.print_nonzero_counts()
            elif self.args.method == 'GMP':
                if self.steps >= (self.args.init_prune_epoch * int(len(self.train_loader)/self.args.world_size) * self.args.multiplier) and \
                        self.steps % self.prune_every_k_steps == 0:
                    print('*************************************Pruning*************************************')
                    self.pruning(self.steps)
            elif self.args.method == 'GMP_uniform':
                if self.steps >= (self.args.init_prune_epoch * int(len(self.train_loader)/self.args.world_size) * self.args.multiplier) and \
                        self.steps % self.prune_every_k_steps == 0:
                    print('*************************************Pruning*************************************')
                    self.pruning_uniform(self.steps)

    def pruning(self, step):
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter =  int((self.args.final_prune_epoch * self.args.multiplier * self.train_loader) / self.prune_every_k_steps)
        ini_iter =  int((self.args.init_prune_epoch * self.args.multiplier * self.train_loader) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')


        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter - 1:
            prune_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3

            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                        1 - prune_decay)

            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))

    def pruning_uniform(self, step):
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter =  int((self.args.final_prune_epoch * int(len(self.train_loader))) / self.prune_every_k_steps)
        ini_iter =  int(self.args.init_prune_epoch * (int(len(self.train_loader))) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter
        print('******************************************************')
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        print('******************************************************')


        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            prune_decay = (1 - ((
                                        curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (
                    1 - prune_decay)

            if curr_prune_rate >= 0.8:
                curr_prune_rate = 1 - (self.total_params * (1-curr_prune_rate) - 0.2 * self.fc_params)/(self.total_params-self.fc_params)

                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        score = torch.flatten(torch.abs(weight))
                        if 'classifier' in name:
                            num_params_to_keep = int(len(score) * 0.2)
                            threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                            acceptable_score = threshold[-1]
                            self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
                        else:
                            num_params_to_keep = int(len(score) * (1 - curr_prune_rate))
                            threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                            acceptable_score = threshold[-1]
                            self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            else:
                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        score = torch.flatten(torch.abs(weight))
                        num_params_to_keep = int(len(score) * (1 - curr_prune_rate))
                        threshold, _ = torch.topk(score, num_params_to_keep, sorted=True)
                        acceptable_score = threshold[-1]
                        self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

            self.apply_mask()

            total_size = 0
            for name, weight in self.masks.items():
                total_size += weight.numel()
            print('Total Model parameters:', total_size)

            sparse_size = 0
            for name, weight in self.masks.items():
                sparse_size += (weight != 0).sum().int().item()

            print('Sparsity after pruning: {0}'.format(
                (total_size-sparse_size) / total_size))


    def add_module(self, module):
        self.modules.append(module)
        self.module = module
        for name, tensor in module.named_parameters():
            if len(tensor.size()) == 2 or len(tensor.size()) == 4:
                if self.args.only_L:
                    if 'large_kernel.LoRA' in name:
                        self.names.append(name)
                        self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)
                else:
                    self.names.append(name)
                    self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).to(self.device)
        # print('Removing biases...')
        # self.remove_weight_partial_name('bias')
        # # print('Removing fisrt layer...')
        # # self.remove_weight_partial_name('conv1.weight')
        # print('Removing 2D batch norms...')
        # self.remove_type(nn.BatchNorm2d, verbose=self.verbose)
        # print('Removing 1D batch norms...')
        # self.remove_type(nn.BatchNorm1d, verbose=self.verbose)

        # if self.args.rm_first:
        #     for name, tensor in module.named_parameters():
        #         if name == 'module.conv1.weight':
        #             self.masks.pop(name)
        #             print(f"pop out {name}")
        self.init(mode=self.args.sparse_init, density=self.args.init_density)


    def is_at_start_of_pruning(self, name):
        if self.start_name is None: self.start_name = name
        if name == self.start_name: return True
        else: return False

    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name+'.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name+'.weight'].shape, self.masks[name+'.weight'].numel()))
            self.masks.pop(name+'.weight')
        else:
            print('ERROR',name)

    def remove_weight_partial_name(self, partial_name, verbose=False):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                if self.verbose:
                    print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape, np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed: self.names.pop(i)
            else: i += 1


    def remove_type(self, nn_type, verbose=False):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)
                    #self.remove_weight_partial_name(name, verbose=self.verbose)


    def apply_mask(self):

        # synchronism masks
        if self.args.distributed:
            self.synchronism_masks()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    if not self.half:
                        tensor.data = tensor.data*self.masks[name]
                        if 'momentum_buffer' in self.optimizer.state[tensor]:
                            self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]
                    else:
                        tensor.data = tensor.data*self.masks[name].half()
                        if name in self.name_to_32bit:
                            tensor2 = self.name_to_32bit[name]
                            tensor2.data = tensor2.data*self.masks[name]

    def adjust_prune_rate(self):
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name not in self.name2prune_rate: self.name2prune_rate[name] = self.prune_rate

                self.name2prune_rate[name] = self.prune_rate

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())
                if sparsity < 0.2:
                    # determine if matrix is relativly dense but still growing
                    expected_variance = 1.0/len(list(self.name2variance.keys()))
                    actual_variance = self.name2variance[name]
                    expected_vs_actual = expected_variance/actual_variance
                    if expected_vs_actual < 1.0:
                        # growing
                        self.name2prune_rate[name] = min(sparsity, self.name2prune_rate[name])

    def truncate_weights(self):

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
                # prune
                new_mask = self.prune_func(self, mask, weight, name)
                removed = self.name2nonzeros[name] - new_mask.sum().item()
                self.total_removed += removed
                self.name2removed[name] = removed
                self.masks[name][:] = new_mask

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                new_mask = self.masks[name].data.byte()
                # growth
                new_mask = self.growth_func(self, name, new_mask, math.floor(self.name2removed[name]), weight)
                # exchanging masks
                # self.masks.pop(name)
                self.masks[name][:] = new_mask.float()

        self.apply_mask()

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']

        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_nonzero_counts(self):
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                num_nonzeros = (mask != 0).sum().item()
                val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
                                                               num_nonzeros / float(mask.numel()))
                print(val)

        print('Prune rate: {0}\n'.format(self.prune_rate))

    def fired_masks_update(self):
        ntotal_fired_weights = 0.0
        ntotal_weights = 0.0
        layer_fired_weights = {}
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                self.fired_masks[name] = self.masks[name].data.byte() | self.fired_masks[name].data.byte()
                ntotal_fired_weights += float(self.fired_masks[name].sum().item())
                ntotal_weights += float(self.fired_masks[name].numel())
                layer_fired_weights[name] = float(self.fired_masks[name].sum().item())/float(self.fired_masks[name].numel())
                # print('Layerwise percentage of the fired weights of', name, 'is:', layer_fired_weights[name])
        total_fired_weights = ntotal_fired_weights/ntotal_weights
        print('The percentage of the total fired weights is:', total_fired_weights)
        return layer_fired_weights, total_fired_weights

    def synchronism_masks(self):

        for name in self.masks.keys():
            torch.distributed.broadcast(self.masks[name], src=0, async_op=False)

