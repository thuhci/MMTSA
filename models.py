from torch import nn
from fusion_classification_network import Fusion_Classification_Network
from transforms import *
from collections import OrderedDict


class MMTSA(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='BNInception', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, midfusion='concat'):
        super(MMTSA, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.num_segments = num_segments
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.midfusion = midfusion
        self.arch = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        self.new_length = OrderedDict()
        if new_length is None:
            for m in self.modality:
                if m == "RGB":
                    self.new_length[m] = 1
                elif m == "Sensor" or m == "AccWatch" or m == "AccPhone" or m == "Orie" or m == 'Gyro':
                    self.new_length[m] = 1
                else: # flow/rgbdif
                    self.new_length[m] = 5
        else:
            
            self.new_length = new_length

        print(("""
Initializing MMTSA with base model: {}.
MMTSA Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        self._prepare_mmtsa()


        is_sensor = any(m=='Sensor' for m in self.modality)
        is_accwatch = any(m=='AccWatch' for m in self.modality)
        is_accphone = any(m=='AccPhone' for m in self.modality)
        is_gyro= any(m=='Gyro' for m in self.modality)
        is_orie = any(m=='Orie' for m in self.modality)
        
        if is_sensor:
            print("Prepare sensor model")
            self.base_model['Sensor'] = self._construct_sensor_model(self.base_model['Sensor'],  self.arch)
            print("Done. Sensor model ready.")
            
        if is_accwatch:
            print("Prepare accwatch model")
            self.base_model['AccWatch'] = self._construct_accwatch_model(self.base_model['AccWatch'],  self.arch)
            print("Done. accwatch model ready.")
        if is_accphone:
            print("Prepare accphone model")
            self.base_model['AccPhone'] = self._construct_accphone_model(self.base_model['AccPhone'],  self.arch)
            print("Done. accphone model ready.")
        if is_gyro:
            print("Prepare gyro model")
            self.base_model['Gyro'] = self._construct_gyro_model(self.base_model['Gyro'],  self.arch)
            print("Done. gyro model ready.")
        if is_orie:
            print("Prepare orie model")
            self.base_model['Orie'] = self._construct_accwatch_model(self.base_model['Orie'],  self.arch)
            print("Done. orie model ready.")

        print('\n')
        
#         print(self.base_model['Sensor'],self.base_model['RGB'])
        

        for m in self.modality:
            self.add_module(m.lower(), self.base_model[m])

    
    def _remove_last_layer(self, arch):
        if arch == 'BNInception':
            # This works only with BNInception.
            for m in self.modality:
                delattr(self.base_model[m], self.base_model[m].last_layer_name)
                for tup in self.base_model[m]._op_list:
                    if tup[0] == self.base_model[m].last_layer_name:
                        self.base_model[m]._op_list.remove(tup)
        elif arch == 'mobilenetv2':
            for m in self.modality:
                self.base_model[m].classifier = nn.Identity()

    def _prepare_mmtsa(self):

        self._remove_last_layer(arch = self.arch)

        self.fusion_classification_net = Fusion_Classification_Network(
            self.feature_dim, self.modality, self.midfusion, self.num_class,
            self.consensus_type, self.before_softmax, self.dropout, self.num_segments)

    def _prepare_base_model(self, base_model):

        if base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = OrderedDict()
            self.input_size = OrderedDict()
            self.input_mean = OrderedDict()
            self.input_std = OrderedDict()

            for m in self.modality:
                self.base_model[m] = getattr(tf_model_zoo, base_model)()
                self.base_model[m].last_layer_name = 'fc'
                self.input_size[m] = 224
                self.input_std[m] = [1]
                if m == 'RGB':
                    self.input_mean[m] = [104, 117, 128]
            self.feature_dim = 1024
        elif base_model == 'mobilenetv2':
            from mobilenet_v2 import mobilenet_v2
            self.base_model = OrderedDict()
            self.input_size = OrderedDict()
            self.input_mean = OrderedDict()
            self.input_std = OrderedDict()
            for m in self.modality:
                self.base_model[m] = mobilenet_v2()
                self.input_std[m] = [0.229, 0.224, 0.225]
                self.input_mean[m] = [0.485, 0.456, 0.406]
                self.input_size[m] = 224
                if m == 'RGB':
                    self.input_mean[m] = [0.485, 0.456, 0.406]
            self.feature_dim = 1280
                
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def freeze_fn(self, freeze_mode):

        if freeze_mode == 'modalities':
            for m in self.modality:
                print('Freezing ' + m + ' stream\'s parameters')
                base_model = getattr(self, m.lower())
                for param in base_model.parameters():
                    param.requires_grad_(False)

        elif freeze_mode == 'partialbn_parameters':
            for mod in self.modality:
                count = 0
                print("Freezing BatchNorm2D parameters except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown parameters update in frozen mode
                            m.weight.requires_grad_(False)
                            m.bias.requires_grad_(False)

        elif freeze_mode == 'partialbn_statistics':
            for mod in self.modality:
                count = 0
                print("Freezing BatchNorm2D statistics except the first one.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        count += 1
                        if count >= 2:
                            # shutdown running statistics update in frozen mode
                            m.eval()
        elif freeze_mode == 'bn_statistics':
            for mod in self.modality:
                print("Freezing BatchNorm2D statistics.")
                base_model = getattr(self, mod.lower())
                for m in base_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        # shutdown running statistics update in frozen mode
                        m.eval()
        else:
            raise ValueError('Unknown mode for freezing the model: {}'.format(freeze_mode))

    def forward(self, input):
        concatenated = []
        # Get the output for each modality
        for m in self.modality:
            if (m == 'Sensor'):
                channel = 6
            elif (m=='RGB',m == 'AccPhone' or m == 'AccWatch' or m == 'Gyro' or m == 'Orie'):
                channel = 3
                
            sample_len = channel * self.new_length[m]
            base_model = getattr(self, m.lower())

            base_out = base_model(input[m].view((-1, sample_len) + input[m].size()[-2:]))
            base_out = base_out.view(base_out.size(0), -1)
            concatenated.append(base_out)
            
            

        output = self.fusion_classification_net(concatenated)
        return output
    
    def _construct_sensor_model(self, base_model, arch):
        
        modules = list(self.base_model['Sensor'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (6 * self.new_length['Sensor'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(6 * self.new_length['Sensor'], conv_layer.out_channels,
                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                     bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach() 

        layer_name = list(container.state_dict().keys())[0][:-7]

        setattr(container, layer_name, new_conv)
        if arch == 'BNInception':
            pool_layer = getattr(self.base_model['Sensor'], 'global_pool')

            new_avg_pooling = nn.AvgPool2d(7, stride=pool_layer.stride, padding=pool_layer.padding)
            setattr(self.base_model['Sensor'], 'global_pool', new_avg_pooling)

        return base_model
        
    def _construct_accphone_model(self, base_model, arch):
        
        modules = list(self.base_model['AccPhone'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length['AccPhone'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(3 * self.new_length['AccPhone'], conv_layer.out_channels,
                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                     bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()

        layer_name = list(container.state_dict().keys())[0][:-7]

        setattr(container, layer_name, new_conv)

        if arch == 'BNInception':
            pool_layer = getattr(self.base_model['AccPhone'], 'global_pool')
            new_avg_pooling = nn.AvgPool2d(7, stride=pool_layer.stride, padding=pool_layer.padding)
            setattr(self.base_model['AccPhone'], 'global_pool', new_avg_pooling)
        return base_model

    def _construct_accwatch_model(self, base_model, arch):
        
        modules = list(self.base_model['AccWatch'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length['AccWatch'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(3 * self.new_length['AccWatch'], conv_layer.out_channels,
                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                     bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach() # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        setattr(container, layer_name, new_conv)

        if arch == 'BNInception':
            pool_layer = getattr(self.base_model['AccWatch'], 'global_pool')
            new_avg_pooling = nn.AvgPool2d(7, stride=pool_layer.stride, padding=pool_layer.padding)
            setattr(self.base_model['AccWatch'], 'global_pool', new_avg_pooling)
            
        return base_model
    
    def _construct_gyro_model(self, base_model, arch):
        
        modules = list(self.base_model['Gyro'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length['Gyro'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(3 * self.new_length['Gyro'], conv_layer.out_channels,
                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                     bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach() # add bias if neccessary

        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        setattr(container, layer_name, new_conv)

        if arch == 'BNInception':
            pool_layer = getattr(self.base_model['Gyro'], 'global_pool')

            new_avg_pooling = nn.AvgPool2d(7, stride=pool_layer.stride, padding=pool_layer.padding)
            setattr(self.base_model['Gyro'], 'global_pool', new_avg_pooling)
        return base_model
            
            
    def _construct_orie_model(self, base_model, arch):
        
        modules = list(self.base_model['Orie'].modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * self.new_length['Orie'], ) + kernel_size[2:]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(3 * self.new_length['Orie'], conv_layer.out_channels,
                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                     bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach() # add bias if neccessary

        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        setattr(container, layer_name, new_conv)

        if arch == 'BNInception':
            pool_layer = getattr(self.base_model['Orie'], 'global_pool')

            new_avg_pooling = nn.AvgPool2d(7, stride=pool_layer.stride, padding=pool_layer.padding)
            setattr(self.base_model['Orie'], 'global_pool', new_avg_pooling)
        
        return base_model
        
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        return scale_size

    def get_augmentation(self):
        augmentation = {}
        if 'RGB' in self.modality:
            augmentation['RGB'] = torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size['RGB'], [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        return augmentation
