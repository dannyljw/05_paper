#%%
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
import torchmetrics
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.quantization import QuantStub, DeQuantStub, fuse_modules,get_default_qconfig
from torch import nn, quantization

from def_224_diet_decoder22_1_t_2 import MobileNetV2 as torch_mobilenet_v2_branch
#seed
pl.seed_everything(777777)

class TrickModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch_mobilenet_v2_branch(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

class QuantizedMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, checkpoint_path=None):
        super().__init__()
        # Initialize the MobileNetV2 branch (replace with your own import)
        self.mobilenet_v2_branch = TrickModel(num_classes=num_classes)
        self._load_checkpoint(checkpoint_path)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.mobilenet_v2_branch(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # Fuse Conv, BN, and ReLU in MobileNetV2
        for module_name, module in self.named_modules():
            if 'ConvBNReLU' in module_name:
                fuse_modules(module, ['0', '1', '2'], inplace=True)
                

    def prepare_qat(self):
        self.fuse_model()
        self.qconfig = quantization.get_default_qat_qconfig('qnnpack')
        quantization.prepare_qat(self, inplace=True)
    
    def _load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)['state_dict']
        # Remove the 'model.mobilenet_v2_branch.' prefix
        new_state_dict = {k.replace('model.mobilenet_v2_branch.', ''): v for k, v in state_dict.items()}
        self.mobilenet_v2_branch.load_state_dict(new_state_dict, strict=False)

class LitQAModel(pl.LightningModule):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.num_classes=1000
        self.dataset_batch_size=32
        self.model = QuantizedMobileNetV2(num_classes=1000, checkpoint_path=checkpoint_path)
        self.model.fuse_model()
        self.model.prepare_qat()
        
        self.val_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
        self.train_acc=torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
    
    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        train_image_preprocess = transforms.Compose([
            transforms.Resize((224,224)), # 80x80
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.RandomCrop((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageNet(
            root='/workspace/shared/datasets/imagenet2012',
            split='train',
            transform=train_image_preprocess)
        return DataLoader(train_dataset, batch_size=self.dataset_batch_size, 
                          shuffle=True, persistent_workers=True, 
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        val_image_preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_dataset = datasets.ImageNet(
            root='/workspace/shared/datasets/imagenet2012',
            split='val',
            transform=val_image_preprocess)
        return DataLoader(val_dataset, batch_size=self.dataset_batch_size,
                          persistent_workers=True, num_workers=8,
                          pin_memory=True)            
    def on_train_epoch_start(self):
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        train_loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('train_loss', train_loss, sync_dist=True)
        self.log('train_acc', self.train_acc(outputs, labels), sync_dist=True)  
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(outputs, labels))
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_train_end(self):
        try:
            self.model.eval()  # 모델을 평가 모드로 설정
            quantization.disable_observer(self.model)
            quant_model = quantization.convert(self.model.to('cpu'))
            torch.save(quant_model, "final_quantized_model.pth")
            # ##############################################################
            # for name, module in quant_model.named_modules():
            #     if type(module) == torch.nn.quantized.modules.conv.Conv2d:
            #         print(name, type(module), module.weight().dtype)
            #         print(module.weight().int_repr())
            #         for name, param in module.named_parameters():
            #             print(name)
            #             print(param)
            #         break
                
            # # torch.jit.save(torch.jit.script(self.model), "final_quantized_model.jit")
            # ###############################################################
            print(f"Model successfully saved!")
        except Exception as e:
            print(f"Failed to save the model: {e}")


import os
checkpoint_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/qat_checkpoint-epoch=12-val_acc=0.56.ckpt"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")


model = LitQAModel(checkpoint_path=checkpoint_path)

# Setup training with model checkpointing
call_back = pl.callbacks.ModelCheckpoint(
    # monitor="val_loss",
    monitor="val_acc",
    mode = "max",
    dirpath="/workspace/3.MyModel/torch_lite_famous_first/checkpoint",
    filename=str(__file__).split('.')[0]+"-{epoch:02d}",
)



# Setup trainer
trainer = pl.Trainer(
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    devices=[0, 1, 2, 3],
    logger=WandbLogger(
        project='mbv2_branch',
        log_model=True, 
        name="t_2_QAT_top5",
    ),
    max_epochs=1,
    callbacks=[call_back],
)

# Validate the model
val_loader = model.val_dataloader()
trainer.validate(model, val_loader)

print("Getting Top-5 accuracy...")
model.to('cuda')
from tqdm.auto import tqdm
# Calculate Top-5 accuracy
@torch.no_grad()
def get_top_accuracy(model, dataloader):
    model.eval()
    top1_acc = torchmetrics.Accuracy(task='multiclass', num_classes=model.num_classes, top_k=1).to(model.device)
    top5_acc = torchmetrics.Accuracy(task='multiclass', num_classes=model.num_classes, top_k=5).to(model.device)

    for batch in tqdm(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        top1_acc.update(outputs, labels)
        top5_acc.update(outputs, labels)
    return top1_acc.compute().item(), top5_acc.compute().item()

top1_accuracy, top5_accuracy = get_top_accuracy(model, val_loader)
print(f"Top-1 Accuracy: {top1_accuracy:.4f}, \tTop-5 Accuracy: {top5_accuracy:.4f}")
