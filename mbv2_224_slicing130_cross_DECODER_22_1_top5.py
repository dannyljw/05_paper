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

# from def_torchvision_mobilenet_v2 import MobileNetV2 as torch_mobilenet_v2

# from def_torchvision_mobilenet_v2_branch import MobileNetV2 as torch_mobilenet_v2_branch
from def_224_diet_decoder22_1 import MobileNetV2 as torch_mobilenet_v2_branch

# Seed
pl.seed_everything(777777)

# Model definition
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_classes = 1000
        self.dataset_batch_size = 64
        self.model = torch_mobilenet_v2_branch(num_classes=self.num_classes)
        
        self.val_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
        self.val_top5_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
            top_k=5
        )
        self.train_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.num_classes,
        )
        
    def train_dataloader(self):
        train_image_preprocess = transforms.Compose([
            transforms.Resize((224, 224)), # 80x80
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.RandomCrop((224, 224)),
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
            transforms.Resize((224, 224)),
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

    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        val_loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_loss', val_loss, sync_dist=True)  
        self.log('val_acc', self.val_acc(outputs, labels), sync_dist=True)
        self.log('val_top5_acc', self.val_top5_acc(outputs, labels), sync_dist=True)
        return val_loss
    
    def on_validation_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-30)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def forward(self, x):
        return self.model(x)

# Load pretrained model and calculate top-5 accuracy
model = LitModel()
checkpoint_path = "/workspace/3.MyModel/torch_lite_famous_first/checkpoint/mbv2_224_slicing130_cross_DECODER_22_1-epoch=193.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path)

# Top-5 accuracy
def get_top5_accuracy(model, dataloader):
    model.eval()
    top5_acc = torchmetrics.Accuracy(task='multiclass', num_classes=model.num_classes, top_k=5).to(model.device)
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            top5_acc.update(outputs, labels)
    return top5_acc.compute().item()

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
    callbacks=[pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="/workspace/3.MyModel/torch_lite_famous_first/checkpoint",
        filename=str(__file__).split('.')[0]+"-{epoch:02d}",
    )],
)

trainer.validate(model)
val_loader = model.val_dataloader()
top5_accuracy = get_top5_accuracy(model, val_loader)
print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
