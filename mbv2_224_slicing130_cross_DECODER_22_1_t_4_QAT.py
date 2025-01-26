#%%
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qat_qconfig
from torch import quantization

# from def_torchvision_mobilenet_v2 import MobileNetV2 as torch_mobilenet_v2
# from def_torchvision_mobilenet_v2_branch import MobileNetV2 as torch_mobilenet_v2_branch
from def_224_diet_decoder22_1_t_4 import MobileNetV2 as torch_mobilenet_v2_branch

# Seed
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
        self.mobilenet_v2_branch = TrickModel(num_classes=num_classes)
        self.mobilenet_v2_branch.load_state_dict(torch.load(checkpoint_path)['state_dict'])
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

class LitQAModel(pl.LightningModule):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.num_classes = 1000
        self.dataset_batch_size = 32
        self.model = QuantizedMobileNetV2(num_classes=1000, checkpoint_path=checkpoint_path)
        self.model.fuse_model()
        self.model.prepare_qat()
        
        self.val_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
        self.train_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
    
    def forward(self, x):
        return self.model(x)
    
    def train_dataloader(self):
        train_image_preprocess = transforms.Compose([
            transforms.Resize((224,224)), 
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
            self.model.eval()  # 모델을 평가 모드로 설정
            quantization.disable_observer(self.model)
            quant_model = quantization.convert(self.model.to('cpu'))
            torch.save(quant_model, "final_quantized_model.pth")
            print(f"Model successfully saved!")
        except Exception as e:
            print(f"Failed to save the model: {e}")

# Setup training with model checkpointing
call_back = pl.callbacks.ModelCheckpoint(
    monitor="val_acc",
    mode="max",
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
        name="qat_training",
    ),
    max_epochs=10,
    callbacks=[call_back],
)

checkpoint_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/mbv2_224_slicing130_cross_DECODER_22_1_t_4-epoch=155.ckpt"

model = LitQAModel(checkpoint_path=checkpoint_path)

trainer.fit(model)
