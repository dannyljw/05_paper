from torch import nn
import torch
# from .utils import load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']

# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }


# class TestModule(nn.Module):
#     def __init__(self) -> None:
        
#         super().__init__()
#         self.dum_param = nn.Parameter(torch.ones(1))
        
#     def forward(self,x):
#         return torch.mul(self.dum_param,x)
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # self.inp = inp
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            # print("add part size")
            # print(x.shape)
            # print(self.conv(x).shape)
            return x + self.conv(x)
        else:
            
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 one_more_setting=None,
                 decoder_residual_setting=None,
                 rest_residual_setting=None,
                #  round_nearest=8,
                round_nearest=4,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 8  # 32->8 
        last_channel = 640  # 1280 -> 160 992-> 800
        

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                #t: expansion factor, c: output channel, n: number of blocks, s: stride
                [4, 4, 1, 1], # 16->8
                [4, 20, 2, 2],
                [4, 64, 3, 2],
                # [4, 128, 3, 1],
                # [4, 160, 3, 2],
                # [4, 320, 1, 1],
            ]
        if one_more_setting is None:
            one_more_setting = [
                # [4, 64, 4, 2] # 64 x 7x7
                [4, 64, 1, 1]
                # [4, 64, 4, 1]
            ]
        if rest_residual_setting is None:
            rest_residual_setting = [
                [4, 60, 1, 1],
                [4, 60, 2, 2],
                [4, 60, 2, 1],
                [4, 60, 1, 1], #320->160
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.features_0 = nn.ModuleList()
        self.features_0_2 = nn.ModuleList()
        self.features_0_3 = nn.ModuleList()
        
        self.features_1 = nn.ModuleList()
        self.features_1_2 = nn.ModuleList()
        self.features_1_3 = nn.ModuleList()
        
        self.features_2 = nn.ModuleList()
        self.features_2_2 = nn.ModuleList()
        self.features_2_3 = nn.ModuleList()
        
        self.features_3 = nn.ModuleList()
        self.features_3_2 = nn.ModuleList()
        self.features_3_3 = nn.ModuleList()
        
        self.features_4 = nn.ModuleList()
        self.features_4_2 = nn.ModuleList()
        self.features_4_3 = nn.ModuleList()
        
        seq_list = [self.features_0, self.features_1, self.features_2, self.features_3, self.features_4]
        
        for A in seq_list:
            # building first layer
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            # A.append(nn.Sequential(ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)))
            A.append((ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)))
            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:
                # print(t,c,n,s)
                output_channel = _make_divisible(c * width_mult, round_nearest)
                inverted_residuals = []
                for i in range(n):
                    stride = s if i == 0 else 1
                    inverted_residuals.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    input_channel = output_channel
                A.extend(inverted_residuals)
        
        seq_3_list = [self.features_0_3, self.features_1_3, self.features_2_3, self.features_3_3, self.features_4_3]
        for D in seq_3_list:
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            for t, c, n, s in one_more_setting:
                # print(t,c,n,s)
                output_channel = _make_divisible(c * width_mult, round_nearest)
                one_more_residuals = []
                for i in range(n):
                    stride = s if i == 0 else 1
                    one_more_residuals.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    input_channel = output_channel
                D.extend(one_more_residuals)
        
        

        seq_2_list = [self.features_0_2, self.features_1_2, self.features_2_2, self.features_3_2, self.features_4_2]
        
        for C in seq_2_list:
            # input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            input_channel = _make_divisible(input_channel * width_mult, round_nearest) #fixed_ input_channel = 64
            using_input_channel = input_channel
            # print("BBBBBBBB iput_channle", using_input_channel)
            # input_channel = 128
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            for t, c, n, s in rest_residual_setting:
                # print("START")
                # print("t,c,n,s",t,c,n,s)
                output_channel = _make_divisible(c * width_mult, round_nearest)
                rest_residuals = []
                for i in range(n):
                    stride = s if i == 0 else 1
                    # print("block_input & output", using_input_channel, output_channel)
                    rest_residuals.append(block(using_input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    using_input_channel = output_channel
                    # print("changed input_channel", using_input_channel)
                    # print()
                C.extend(rest_residuals)
        
            # building last several layers
            C.append(nn.Sequential(ConvBNReLU(using_input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)))
            # print("LLLLLLLLLLLLLLLLLLLLL",self.last_channel)
        # print(id(A[0]), id(A[1]), id(A[2]))
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x0 = x[:, :, :130, :130]
        x1 = x[:, :, :130, 94:224]
        x2 = x[:, :, 94:224, :130]
        x3 = x[:, :, 94:224, 94:224]
        x4 = x[:, :, 47:177, 47:177]
#-------------------------------        
        # Forward pass for each branch
        for layer in self.features_0:
            # print("First input x0", x0.shape)
            x0 = layer(x0)
            # print("First output x0", x0.shape)
            # print()
        for layer in self.features_0_3:
            # print("Second input x0", x0.shape)
            x0_3 = layer(x0)
            # print("Second output x0_3", x0_3.shape)
            # print()
        x0_1 = x0 + x0_3 # 64 x 17x17
        # print("ADD x0 + x0_3 = x0_1", x0.shape, x0_3.shape, x0_1.shape)
        # print()
        for layer in self.features_0_2:
            # input -> 64 x 17x17
            # print("Fourth input x0_1", x0_1.shape)
            # print("layer.", layer.inp)
            x0_1 = layer(x0_1)
            # print("Fourth x0_1.shape", x0_1.shape)
            # print()
#-------------------------------
        for layer in self.features_1:
            x1 = layer(x1)
        x1_3 =x1
        for layer in self.features_1_3:
            # x1_3 = layer(x1)
            x1_3 = layer(x1_3)
        # for layer in self.features_1_1:
            # x1_1 = layer(x1_3)
        x1_1 = x1 + x1_3 # 64 x 17x17
            # print("x1_1 x1_3 shape ", x1_1.shape, x1_3.shape)
        for layer in self.features_1_2:# input -> 64 x 17x17
            # print("before x1_1.shape", x1_1.shape)
            x1_1 = layer(x1_1)
            # print("after x1_1.shape", x1_1.shape)
#-------------------------------          
        for layer in self.features_2:
            x2 = layer(x2)
        x2_3 = x2
        for layer in self.features_2_3:
            x2_3 = layer(x2_3)
        x2_1 = x2 + x2_3 # 64 x 17x17
        for layer in self.features_2_2:# input -> 64 x 17x17
            x2_1 = layer(x2_1)
#-------------------------------        
         
        for layer in self.features_3:
            x3 = layer(x3)
        x3_3 = x3
        for layer in self.features_3_3:
            x3_3 = layer(x3_3)
        x3_1 = x3 + x3_3 # 64 x 17x17
        for layer in self.features_3_2:# input -> 64 x 17x17
            x3_1 = layer(x3_1)
 #-------------------------------       
        for layer in self.features_4:
            x4 = layer(x4)
        x4_3 = x4
        for layer in self.features_4_3:
            x4_3 = layer(x4_3)
        x4_1 = x4 + x4_3 # 64 x 17x17
        for layer in self.features_4_2:# input -> 64 x 17x17
            x4_1 = layer(x4_1)
            
           

            
        # Global average pooling and flattening
        # print(x4.shape)
        x0 = nn.functional.adaptive_avg_pool2d(x0_1, 1).reshape(x0_1.shape[0], -1)
        x1 = nn.functional.adaptive_avg_pool2d(x1_1, 1).reshape(x1_1.shape[0], -1)
        x2 = nn.functional.adaptive_avg_pool2d(x2_1, 1).reshape(x2_1.shape[0], -1)
        x3 = nn.functional.adaptive_avg_pool2d(x3_1, 1).reshape(x3_1.shape[0], -1)
        x4 = nn.functional.adaptive_avg_pool2d(x4_1, 1).reshape(x4_1.shape[0], -1)
        
        # Adding the flattened tensors
        add_x = torch.add(x0, x1)
        add_x = torch.add(add_x, x2)
        add_x = torch.add(add_x, x3)
        add_x = torch.add(add_x, x4)
        
        
        x = self.classifier(add_x)
        return x

    
from torchsummary import summary
summary_model = MobileNetV2(num_classes=1000)
summary(model=summary_model, input_size=(3,224,224), batch_size=64, device='cpu')

# from def_peak_mem import *
# top3_values, fc_params = get_top3_values(summary_model, (3, 224, 224))