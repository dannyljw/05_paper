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

import pyvww
from tqdm.auto import tqdm
from def_peak_mem import *
# from def_torchvision_mobilenet_v2 import MobileNetV2 as torch_mobilenet_v2

# from def_torchvision_mobilenet_v2_branch import MobileNetV2 as torch_mobilenet_v2_branch
from def_224_diet_decoder22_1_vwv import MobileNetV2 as torch_mobilenet_v2_branch
#seed
pl.seed_everything(777777)


# from torchsummary import summaryckp

# summary_model = torch_mobilenet_v2_branch(num_classes=2)
# summary(model=summary_model, input_size=(3,130,130), batch_size=64, device='cpu') 
# #%%

class TrickModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch_mobilenet_v2_branch(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class QuantizedMobileNetV2(nn.Module):
    def __init__(self, num_classes=2, checkpoint_path=None):
        super().__init__()
        # Initialize the MobileNetV2 branch (replace with your own import)
        self.mobilenet_v2_branch = TrickModel(num_classes=num_classes)
        self.mobilenet_v2_branch.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        # checkpoint_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/mbv2_224_slicing130_cross_DECODER_22_1-epoch=193.ckpt"
        # model = LitQAModel.load_from_checkpoint(checkpoint_path)
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



class LitQAModel(pl.LightningModule):
    def __init__(self, checkpoint_path = None):
        super().__init__()
        self.num_classes=2
        self.dataset_batch_size=64
        self.model = torch_mobilenet_v2_branch(num_classes=2, checkpoint_path=checkpoint_path)
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
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root='/workspace/shared/datasets/vww/all2014',
            annFile='/workspace/shared/datasets/new-vww/annotations/instances_train.json',
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
        val_dataset = pyvww.pytorch.VisualWakeWordsClassification(
            root='/workspace/shared/datasets/vww/all2014',
            annFile='/workspace/shared/datasets/new-vww/annotations/instances_val.json',
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

    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', val_loss, sync_dist=True)  
        self.log('val_acc', self.val_acc(outputs, labels), sync_dist=True)
        return val_loss
    def on_validation_epoch_end(self):
        pass
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-30)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    def on_train_end(self):
        try:
            self.model.eval()
            quantization.disable_observer(self.model)
            quant_model = quantization.convert(self.model.to('cpu'))
            torch.save(quant_model, "final_quantized_model_VWW.pth")
            print(f'model successfullly saved!')
        except Exception as e:
            print(f'Error: {e}')
            print(f'model not saved!')



 
# model = LitModel()
# top3_values, fc_params = get_top3_values(model, (3, 224, 224))
# print(top3_values)
# print("Fully Connected Layer Param", fc_params)
# exit()
call_back = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath="/workspace/3.MyModel/torch_lite_famous_first/checkpoint",
    filename=str(__file__).split('.')[0]+"-{epoch:02d}",
)
trainer = pl.Trainer(
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    devices=[0, 1, 2, 3],
    logger=WandbLogger(
        project='mbv2_branch',
        log_model=True, 
        name=str(__file__).split('.')[0],
    ),
    max_epochs=500,
    callbacks=[call_back],
    # code for training from checkpoint.
    
)
checkpoint_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/mbv2_224_slicing130_cross_DECODER_22_1_vwv-epoch=79.ckpt"
model = LitQAModel(checkpoint_path = checkpoint_path)
trainer.fit(model)
