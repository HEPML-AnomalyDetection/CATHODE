import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import flows as fnn
import mdn


class DensityEstimator:
    class Unknown(Exception):
        """ Error to raise for unkown DensityEstimator model """

    @classmethod
    def get_all_subclasses(cls):
        """ Get all the subclasses recursively of this class. """
        for subclass in cls.__subclasses__():
            yield from subclass.get_all_subclasses()
            yield subclass

    @classmethod
    def _get_name(cls, name):
        return name.upper()

    @classmethod
    def _parse_yaml(cls, filename):
        with open(filename, 'r') as stream:
            cls.params = yaml.safe_load(stream)

        name = cls.params['ModelType']
        if name != 'MDN':
            if cls.params['Transform'] == 'RQS':
                name = name + '_RQS'

        return name

    def __new__(cls, *args, **kwargs):
        name = cls._parse_yaml(args[0])
        name = cls._get_name(name)
        for subclass in cls.get_all_subclasses():
            if subclass.name == name:
                # Using "object" base class methods avoid recursion here.
                return object.__new__(subclass)
        raise DensityEstimator.Unknown(f'Unknown model "{name}" requested')

    def __init__(self, filename, eval_mode=False, load_path=None,
                 device=torch.device("cpu"), verbose=False):
        # with open(filename, 'r') as stream:
        #     params = yaml.safe_load(stream)

        self.bound = False

        self.build(self.params, eval_mode, load_path, device, verbose)

    def build(self, params, eval_mode, load_path, device, verbose):
        """
        Used for building a flow based density estimator model
        from a yaml config file.
        """
        modules = []
        for i in range(params['num_blocks']):
            self.build_block(i, modules, params)

        if self.bound:
            self.model = fnn.FlowSequentialUniformBase(*modules)
            # modules += [fnn.InfiniteToFinite(to_finite=False)]
        else:
            self.model = fnn.FlowSequential(*modules)

        self.model = fnn.FlowSequential(*modules)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        # Workaround bug in flow.py
        self.model.num_inputs = params['num_inputs']

        self.finalize_build(params, eval_mode, device, verbose, load_path)

    def load_model(self, load_path):
        if load_path is not None:
            print(f"Loading model parameters from {load_path}")
            self.model.load_state_dict(torch.load(load_path,
                                                  map_location='cpu'))

    def build_optimizer(self, params):
        optimizer = optim.__dict__[params['optimizer']['name']]
        optimizer_kwargs = params['optimizer']
        del optimizer_kwargs['name']
        self.optimizer = optimizer(self.model.parameters(),
                                   **optimizer_kwargs)

    def finalize_build(self, params, eval_mode, device, verbose, load_path):
        self.model.to(device)
        if verbose:
            print(self.model)
        total_parameters = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        print(f"DensityEstimator has {total_parameters} parameters")

        self.load_model(load_path)

        # Get the requested for optimizer
        self.build_optimizer(params)

        if eval_mode:
            self.model.eval()

    def build_block(self, modules, params):
        raise NotImplementedError


class DE_MAF(DensityEstimator):
    name = 'MAF'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_block(self, i, modules, params):
        modules += [
            fnn.MADE(params['num_inputs'], params['num_hidden'],
                     params['num_cond_inputs'],
                     act=params['activation_function'],
                     pre_exp_tanh=params['pre_exp_tanh']),
            ]
        if params['batch_norm']:
            modules += [fnn.BatchNormFlow(params['num_inputs'],
                                          momentum=params['batch_norm_momentum'])]
        modules += [fnn.Reverse(params['num_inputs'])]
