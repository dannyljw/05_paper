#%%
import torch
from def_224_diet_decoder22_1_NEW_QAT import MobileNetV2 as torch_mobilenet_v2_branch
from torch.quantization import QuantStub, DeQuantStub
from torch import nn

# QuantizedMobileNetV2 및 TrickModel 클래스 정의
class TrickModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch_mobilenet_v2_branch(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

class QuantizedMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, checkpoint_path=None):
        super().__init__()
        self.mobilenet_v2_branch = TrickModel(num_classes=num_classes)
        # self.mobilenet_v2_branch.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.mobilenet_v2_branch(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for module_name, module in self.named_modules():
            if 'ConvBNReLU' in module_name:
                fuse_modules(module, ['0', '1', '2'], inplace=True)

    def prepare_qat(self):
        self.fuse_model()
        self.qconfig = get_default_qat_qconfig('qnnpack')
        quantization.prepare_qat(self, inplace=True)

# %%
file_path = "/workspace/3.MyModel/05_paper/final_quantized_model.pth"
# model = torch.load(file_path, map_location='cpu')  # 전체 모델 로드
model = torch.load(file_path)  # 전체 모델 로드

# %%
print(model.mobilenet_v2_branch.model)

# 모델의 모든 속성을 확인하여 weight와 bias 출력
print("Model's weights and biases:")
for name in dir(model):
    attr = getattr(model, name)
    if isinstance(attr, torch.Tensor):
        print("yes isinstance?")
        print(f"{name}: {attr.shape}")
    elif isinstance(attr, torch.nn.Module):
        print("no isinstance?")
        # Check if the submodule has weight and bias attributes
        if hasattr(attr, 'weight') and isinstance(attr.weight, torch.Tensor):
            print(f"{name}.weight: {attr.weight.shape}")
        if hasattr(attr, 'bias') and isinstance(attr.bias, torch.Tensor):
            print(f"{name}.bias: {attr.bias.shape}")
