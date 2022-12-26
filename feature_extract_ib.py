from webbrowser import get
from datasets.PlastSurg import PlastSurgDataset
import argparse
from utils.config import get_config_ibfocal
from utils.utils import torch_fix_seed
from utils.losses import IB_FocalLoss, AutomaticWeightedLoss
import yaml
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pickle
import timm
from pathlib import Path
import wandb
import logging
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import torchmetrics



class TwoHeadResNet50(nn.Module):
    def __init__(self,hparams):
        super(TwoHeadResNet50, self).__init__()
        self.backbone = timm.create_model(hparams.model_name,pretrained=hparams.pretrained, num_classes=0)
        self.in_features = self.backbone.num_features
        self.out_features = hparams.out_features
        self.fc_phase = nn.Linear(self.in_features,hparams.out_features)
        self.fc_tool = nn.Linear(self.in_features, hparams.tool_features)


    def forward(self,x,train,y_phase):
        out_stem = self.backbone(x)
        tool = self.fc_tool(out_stem)
        phase = self.fc_phase(out_stem)

        return out_stem, phase, tool

class FeatureExtraction(LightningModule):
    def __init__(self,hparams,model,dataset):
        super(FeatureExtraction, self).__init__()
        self.hparams__ = hparams
        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.num_tasks = self.hparams__.num_tasks
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.bce_loss = nn.BCEWithLogitsLoss() # tool loss
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.dataset.phase_weights).float()) # phase loss
        self.ib_focal_loss = IB_FocalLoss(weight=torch.from_numpy(self.dataset.phase_weights).float().cuda(self.hparams__.gpus[0]),num_classes=self.hparams__.out_features)
        self.aw_loss = AutomaticWeightedLoss(2)
        self.sig_f = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.current_video_idx = self.dataset.df["test"].video_idx.min()
        self.init_metrics()

        #store model
        self.current_stems = []
        self.current_phase_labels = []
        self.current_p_phases = []
        self.current_p_tool = []
        self.len_test_data = len(self.dataset.data["test"])
        self.model = model
        self.best_metrics_high = {"val_acc_phase" : 0}
        self.test_acc_per_video = {}
        self.pickle_path = None
    
    def init_metrics(self):
        self.train_acc_phase = torchmetrics.Accuracy(task='multiclass',num_classes=self.hparams__.out_features)
        self.val_acc_phase = torchmetrics.Accuracy(task='multiclass',num_classes=self.hparams__.out_features)
        self.val_f1_phase = torchmetrics.F1Score(num_classes=self.hparams__.out_features,task='multiclass',average='macro')
        self.test_acc_phase = torchmetrics.Accuracy(task='multiclass',num_classes=self.hparams__.out_features)
        self.test_f1_phase = torchmetrics.F1Score(num_classes=self.hparams__.out_features,task='multiclass',average='macro')
        self.test_acc_tool = torchmetrics.Accuracy(task='multilabel',num_labels=self.hparams__.tool_features)
        self.test_f1_tool = torchmetrics.F1Score(num_labels=self.hparams__.tool_features, task='multilabel')
        self.train_acc_tool = torchmetrics.Accuracy(task='multilabel',num_labels=self.hparams__.tool_features)
        self.train_f1_tool = torchmetrics.F1Score(num_labels=self.hparams__.tool_features, task='multilabel')
        self.val_acc_tool = torchmetrics.Accuracy(task='multilabel',num_labels=self.hparams__.tool_features)
        self.val_f1_tool = torchmetrics.F1Score(num_labels=self.hparams__.tool_features,task='multilabel')

    def forward(self,x,train,y_phase=False):
        stem, phase, tool = self.model.forward(x,train,y_phase)
        return stem, phase, tool
    

    def loss_phase_tool(self, stem, p_phase, p_tool, labels_phase, labels_tool, num_tasks):
        if super().current_epoch >= self.hparams__.ib_start_epoch:
            loss_phase = self.ib_focal_loss(p_phase,labels_phase,stem)
        else:
            loss_phase = self.ce_loss(p_phase, labels_phase)
        if num_tasks == 1:
            return loss_phase
        if num_tasks == 0:
            labels_tool = torch.stack(labels_tool, dim=1)
            loss_tools = self.bce_loss(p_tool, labels_tool.data.float())
            return loss_tools
        else:
            labels_tool = torch.stack(labels_tool, dim=1)
            loss_tools = self.bce_loss(p_tool, labels_tool.data.float())
            loss = self.aw_loss(loss_phase,loss_tools)
            return loss


    def training_step(self, batch, batch_idx):
        x, y_phase, y_tool = batch
        stem, p_phase, p_tool = self.forward(x,True,y_phase)
        loss = self.loss_phase_tool(stem, p_phase, p_tool, y_phase, y_tool, self.num_tasks)
        # acc_phase, acc_tool, loss
        if self.num_tasks == 2 or self.num_tasks == 0:
            self.train_acc_tool(p_tool, torch.stack(y_tool, dim=1))
            self.log("train_acc_tool", self.train_acc_tool, on_epoch=True, on_step=False)
            self.train_f1_tool(p_tool, torch.stack(y_tool, dim=1))
            self.log("train_f1_tool", self.train_f1_tool, on_epoch=True, on_step=False)
        self.train_acc_phase(p_phase, y_phase)
        self.log("train_acc_phase", self.train_acc_phase, on_epoch=True, on_step=True)
        self.log("loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y_phase, y_tool = batch
        stem, p_phase, p_tool = self.forward(x,True,y_phase)
        loss = self.loss_phase_tool(stem, p_phase, p_tool, y_phase, y_tool, self.num_tasks)
        # acc_phase, acc_tool, loss
        if self.num_tasks == 2 or self.num_tasks == 0:
            self.val_acc_tool(p_tool, torch.stack(y_tool, dim=1))
            self.log("val_acc_tool", self.val_acc_tool, on_epoch=True, on_step=False)
            self.val_f1_tool(p_tool, torch.stack(y_tool, dim=1))
            self.log("val_f1_tool", self.val_f1_tool, on_epoch=True, on_step=False)
        self.val_acc_phase(p_phase, y_phase)
        self.log("val_acc_phase", self.val_acc_phase, on_epoch=True, on_step=False)
        self.val_f1_phase(p_phase, y_phase)
        self.log("val_f1_phase", self.val_f1_phase, on_epoch=True, on_step=False)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

    def get_phase_acc(self, true_label, pred):
        pred = torch.FloatTensor(pred)
        pred_phase = torch.softmax(pred, dim=1)
        labels_pred = torch.argmax(pred_phase, dim=1).cpu().numpy()
        cm = ConfusionMatrix(
            actual_vector=true_label,
            predict_vector=labels_pred,
        )
        return cm.Overall_ACC, cm.PPV, cm.TPR, cm.classes, cm.F1_Macro

    def save_to_drive(self, vid_index):
        vid_index = str(vid_index).zfill(2)
        acc, ppv, tpr, keys, f1 = self.get_phase_acc(self.current_phase_labels,
                                                     self.current_p_phases)
        save_path = self.pickle_path / f"{self.hparams__.fps_sampling_test}fps_TF={str(self.hparams__.add_tool_feats)}_{self.hparams__.name}"
        save_path.mkdir(exist_ok=True)
        save_path_txt = save_path / f"video_{vid_index}_{self.hparams__.fps_sampling_test}fps_acc.txt"
        save_path_vid = save_path / f"video_{vid_index}_{self.hparams__.fps_sampling_test}fps.pkl"

        with open(save_path_txt, "w") as f:
            f.write(
                f"vid: {vid_index}; acc: {acc}; ppv: {ppv}; tpr: {tpr}; keys: {keys}; f1: {f1}"
            )
            self.test_acc_per_video[vid_index] = acc
            print(
                f"save video {vid_index} | acc: {acc:.4f} | f1: {f1}"
            )
        if self.hparams__.add_tool_feats:
            with open(save_path_vid, 'wb') as f:
                pickle.dump([
                    #np.asarray(self.current_stems),
                    np.concatenate([np.asarray(self.current_stems),np.asarray(self.current_p_tool)],1),
                    np.asarray(self.current_p_phases),
                    np.asarray(self.current_phase_labels),
                ], f)
        else:
            with open(save_path_vid, 'wb') as f:
                pickle.dump([
                    np.asarray(self.current_stems),
                    #np.concatenate([np.asarray(self.current_stems),np.asarray(self.current_p_tool)],1),
                    np.asarray(self.current_p_phases),
                    np.asarray(self.current_phase_labels),
                ], f)            

    def test_step(self, batch, batch_idx):
        x, y_phase, (vid_idx, image_path, img_index, tool_skinmarker, tool_syringe,
               tool_scalpel, tool_scissors,tool_electric_cautery, tool_structure_and_needle, tool_bipolar_forceps
               ) = batch
        y_tool = (tool_skinmarker, tool_syringe,
               tool_scalpel, tool_scissors,tool_electric_cautery, tool_structure_and_needle, tool_bipolar_forceps)
        vid_idx_raw = vid_idx.cpu().numpy()
        with torch.no_grad():
            stem, y_hat, y_hat_tool = self.forward(x,False)
            y_hat = self.softmax(y_hat)
            y_hat_tool = self.sigmoid(y_hat_tool)
        self.test_acc_phase(y_hat, y_phase)
        self.test_f1_phase(y_hat,y_phase)
        self.test_acc_tool(y_hat_tool, torch.stack(y_tool,dim=1))
        self.test_f1_tool(y_hat_tool, torch.stack(y_tool,dim=1))
        #self.log("test_acc_phase", self.test_acc_phase, on_epoch=True, on_step=True)
        vid_idxs, indexes = np.unique(vid_idx_raw, return_index=True)
        vid_idxs = [int(x) for x in vid_idxs]
        index_next = len(vid_idx) if len(vid_idxs) == 1 else indexes[1]
        for i in range(len(vid_idxs)):
            vid_idx = vid_idxs[i]
            index = indexes[i]
            if vid_idx != self.current_video_idx:
                self.save_to_drive(self.current_video_idx)
                self.current_stems = []
                self.current_phase_labels = []
                self.current_p_phases = []
                self.current_p_tool = []
                if len(vid_idxs) <= i + 1:
                    index_next = len(vid_idx_raw)
                else:
                    index_next = indexes[i+1]  # for the unlikely case that we have 3 phases in one batch
                self.current_video_idx = vid_idx
            y_hat_numpy = np.asarray(y_hat.cpu()).squeeze()
            self.current_p_phases.extend(
                np.asarray(y_hat_numpy[index:index_next, :]).tolist())
            self.current_stems.extend(
                stem[index:index_next, :].cpu().detach().numpy().tolist())
            y_phase_numpy = y_phase.cpu().numpy()
            self.current_phase_labels.extend(
                np.asarray(y_phase_numpy[index:index_next]).tolist())
            y_tool_numpy = y_hat_tool.cpu().numpy()
            self.current_p_tool.extend(
                np.asarray(y_tool_numpy[index:index_next]).tolist())

        if (batch_idx + 1) * self.hparams__.batch_size >= self.len_test_data:
            self.save_to_drive(vid_idx)
            print(f"Finished extracting all videos...")


    def test_epoch_end(self, outputs):
        self.log("test_acc_phase", float(self.test_acc_phase.compute()))
        self.log("test_f1_phase", float(self.test_f1_phase.compute()))
        self.log("test_acc_tool", float(self.test_acc_tool.compute()))
        self.log("test_f1_tool", float(self.test_f1_tool.compute()))
    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.RAdam(self.parameters(),
                               lr=self.hparams__.learning_rate,
                               weight_decay=1e-5
                               )
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer]  #, [scheduler]   


    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        generator = torch.Generator()
        if split == "test":
            should_shuffle = False
        else:
            should_shuffle = True
        print(f"split: {split} - shuffle: {should_shuffle}")
        worker = self.hparams__.num_workers
        if split == "test" and self.hparams__.test_extract:
            print(
                "worker set to 0 due to test"
            )  # otherwise for extraction the order in which data is loaded is not sorted e.g. 1,2,3,4, --> 1,5,3,2
            worker = 0

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams__.batch_size,
            shuffle=should_shuffle,
            num_workers=worker,
            pin_memory=True,
            worker_init_fn=torch_fix_seed,
            generator=generator
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info("training data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info("validation data loader called - size: {}".format(
            len(dataloader.dataset)))
        return dataloader


    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info("test data loader called  - size: {}".format(
            len(dataloader.dataset)))
        print(f"starting video idx for testing: {self.current_video_idx}")
        self.set_export_pickle_path()
        return dataloader


    def set_export_pickle_path(self):
        self.pickle_path = self.hparams__.feature_output_path + "/pickle_export"
        self.pickle_path = Path(self.pickle_path)
        self.pickle_path.mkdir(exist_ok=True)
        print(f"setting export pickle path: {self.pickle_path}")

# train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    config_name = args.config
    config_path = '/media/aolab/untitiled/workspace/PlastSurg/config/'+config_name
    #config_path = '/media/aolab/untitiled/workspace/PlastSurg/config/config_feature_extract.yaml'
    
    torch_fix_seed()
    
    d = {}
    with open(config_path, mode="r") as f:
        d = yaml.load(f)
    hyperparams = get_config_ibfocal(config_path)
    if hyperparams.wandb:
        wandb.init(
            project="PlastSurg",
            name=hyperparams.name ,
            config=d
        )
    dataset = PlastSurgDataset(hyperparams) 
    model = TwoHeadResNet50(hyperparams)
    module = FeatureExtraction(hyperparams,model,dataset)
    logging.disable(logging.WARNING)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{hyperparams.feature_output_path}/checkpoints/",
        save_top_k=hyperparams.save_top_k,
        verbose=True,
        monitor=hyperparams.early_stopping_metric,
        mode='max',
        filename=hyperparams.name

    )
    early_stop_callback = EarlyStopping(
        monitor=hyperparams.early_stopping_metric,
        min_delta=0.00,
        patience=50,
        mode='max')
    tb_logger = TensorBoardLogger(hyperparams.feature_output_path, name='tb')
    if hyperparams.wandb:
        wandb_logger = WandbLogger(name = hyperparams.name, project="PlastSurg")
    if hyperparams.wandb == True:
        loggers = [wandb_logger]
    else:
        loggers = []
    trainer = Trainer(
        #fast_dev_run=True,
        accelerator='gpu',
        devices=hyperparams.gpus,
        logger=loggers,
        min_epochs=hyperparams.min_epocks,
        max_epochs=hyperparams.max_epocks,
        callbacks=[early_stop_callback,checkpoint_callback],
        num_sanity_val_steps=hyperparams.num_sanity_val_steps,
        deterministic = True, 
    )
    
    if hyperparams.train:
        trainer.fit(module)
        torch.save(module.model.state_dict(),hyperparams.model_path+'/'+hyperparams.name+'.pth')
    module.model.load_state_dict(torch.load(hyperparams.model_path+'/'+hyperparams.name+'.pth'))
    trainer.test(module)