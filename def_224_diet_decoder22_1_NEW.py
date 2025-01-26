from torch import nn
import torch

__all__ = ['MobileNetV2', 'mobilenet_v2']

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
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
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
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
                 round_nearest=4,
                 block=None,
                 norm_layer=None):
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 8 #in : 130x130x3 out : 65x65x8
        last_channel = 640
        
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [6, 4, 1, 1], # in : 65x65x8 out : 65x65x4
                [6, 20, 2, 2], # in_0 : 65x65x4 out_0 : 33x33x20 # in_1 : 33x33x20 out_1 : 33x33x20
                [6, 64, 3, 2], # in_0 : 33x33x20 out_0 : 17x17x64 # int_1 : 17x17x64 out_1 : 17x17x64 # in_2 : 17x17x64 out_2 : 17x17x64
            ]
        if one_more_setting is None:
            one_more_setting = [
                [6, 64, 1, 1] # in_0 : 17x17x64 out_0 : 17x17x64
            ]
        if rest_residual_setting is None:
            rest_residual_setting = [
                [6, 60, 1, 1], #in_0 : 17x17x64 out_0 : 17x17x60
                [6, 60, 2, 2], #in_0 : 17x17x60 out_0 : 9x9x60 # in_1 : 9x9x60 out_1 : 9x9x60
                [6, 60, 2, 1],
                [6, 60, 1, 1],
            ]

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
        
        # Initialize input_channel once before loop
        initial_input_channel = input_channel
        
        for A in seq_list:
            input_channel = _make_divisible(initial_input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            A.append((ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)))
            for t, c, n, s in inverted_residual_setting:
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
                output_channel = _make_divisible(c * width_mult, round_nearest)
                one_more_residuals = []
                for i in range(n):
                    stride = s if i == 0 else 1
                    one_more_residuals.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    input_channel = output_channel
                D.extend(one_more_residuals)
        
        seq_2_list = [self.features_0_2, self.features_1_2, self.features_2_2, self.features_3_2, self.features_4_2]
        
        for C in seq_2_list:
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            using_input_channel = input_channel
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            for t, c, n, s in rest_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                rest_residuals = []
                for i in range(n):
                    stride = s if i == 0 else 1
                    rest_residuals.append(block(using_input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    using_input_channel = output_channel
                C.extend(rest_residuals)
        
            C.append(nn.Sequential(ConvBNReLU(using_input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

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

        for layer in self.features_0:
            x0 = layer(x0)
        for layer in self.features_0_3:
            x0_3 = layer(x0)
        x0_1 = x0 + x0_3
        for layer in self.features_0_2:
            x0_1 = layer(x0_1)

        for layer in self.features_1:
            x1 = layer(x1)
        x1_3 = x1
        for layer in self.features_1_3:
            x1_3 = layer(x1_3)
        x1_1 = x1 + x1_3
        for layer in self.features_1_2:
            x1_1 = layer(x1_1)

        for layer in self.features_2:
            x2 = layer(x2)
        x2_3 = x2
        for layer in self.features_2_3:
            x2_3 = layer(x2_3)
        x2_1 = x2 + x2_3
        for layer in self.features_2_2:
            x2_1 = layer(x2_1)

        for layer in self.features_3:
            x3 = layer(x3)
        x3_3 = x3
        for layer in self.features_3_3:
            x3_3 = layer(x3_3)
        x3_1 = x3 + x3_3
        for layer in self.features_3_2:
            x3_1 = layer(x3_1)

        for layer in self.features_4:
            x4 = layer(x4)
        x4_3 = x4
        for layer in self.features_4_3:
            x4_3 = layer(x4_3)
        x4_1 = x4 + x4_3
        for layer in self.features_4_2:
            x4_1 = layer(x4_1)

        x0 = nn.functional.adaptive_avg_pool2d(x0_1, 1).reshape(x0_1.shape[0], -1)
        x1 = nn.functional.adaptive_avg_pool2d(x1_1, 1).reshape(x1_1.shape[0], -1)
        x2 = nn.functional.adaptive_avg_pool2d(x2_1, 1).reshape(x2_1.shape[0], -1)
        x3 = nn.functional.adaptive_avg_pool2d(x3_1, 1).reshape(x3_1.shape[0], -1)
        x4 = nn.functional.adaptive_avg_pool2d(x4_1, 1).reshape(x4_1.shape[0], -1)
        
        add_x = torch.add(x0, x1)
        add_x = torch.add(add_x, x2)
        add_x = torch.add(add_x, x3)
        add_x = torch.add(add_x, x4)
        
        x = self.classifier(add_x)
        return x

from torchsummary import summary
summary_model = MobileNetV2(num_classes=1000)
summary(model=summary_model, input_size=(3,224,224), batch_size=64, device='cpu')
