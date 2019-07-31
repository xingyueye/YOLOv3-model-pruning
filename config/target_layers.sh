from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

# from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

ori_layers = {0:32, 3:64, 6:32, 9:64, 12:128, 15:64, 18:128, 21:64, 24:128, 27:256, 30:128, 33:256, 36:128,
              39:256, 42:128, 45:256, 48:128, 51:256, 54:128, 57:256, 60:128, 63:256, 66:128, 69:256, 72:128,
              75:256, 78:512, 81:256, 84:512, 87:256, 90:512, 93:256, 96:512, 99:256, 102:512, 105:256, 108:512,
              111:256, 114:512, 117:256, 120:512, 123:256,126:512, 129:1024, 132:512, 135:1024, 138:512, 141:1024,
              144:512, 147:1024, 150:512, 153:1024, 156:512, 159:1024, 162:512, 165:1024, 168:512, 171:1024,
              174:255, 177:256, 180:256, 183:512, 186:256, 189:512, 192:256, 195:512, 198:255, 201:128, 204:128,
              207:256, 210:128, 213:256, 216:128, 219:256, 222:255}
target_layers = {0:32, 3:64, 6:32, 9:48,
                 12:112, 15:64, 18:128,
                 21:64, 24:128,
                 27:256, 30:128, 33:256,
                 36:128, 39:256,
                 42:128, 45:256,
                 48:128, 51:256,
                 54:128, 57:256,
                 60:128, 63:256,
                 66:128, 69:256,
                 72:128, 75:256,
                 78:512, 81:256, 84:512,
                 87:256, 90:512,
                 93:256, 96:512,
                 99:256, 102:512,
                 105:256, 108:512,
                 111:256, 114:512,
                 117:256, 120:512,
                 123:256,126:512,
                 129:1024, 132:512, 135:1024,
                 138:512, 141:1024,
                 44:512, 147:1024,
                 150:512, 153:1024,
                 # 156:512, 159:1024, 162:512, 165:1024, 168:512, 171:1024, 174:255,
                 # 177:256,
                 # 180:256, 183:512, 186:256, 189:512, 192:256, 195:512, 198:255,
                 # 201:128,
                 # 204:128, 207:256, 210:128, 213:256, 216:128, 219:256, 222:255
                 }

class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(weight_torch.size()[0])
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #print("ori_num: %d , target_num: %d"%(weight_torch.size()[0],filter_pruned_num ))
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            #kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in filter_index:
                codebook[x] = 0
            # print("filter codebook done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        # skip_list = [i for i in range(153, 225, 3)]
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
            if len(item.data.size()) == 4 and index in target_layers:
                self.compress_rate[index] = float(target_layers[index])/(item.data.size()[0])
                # self.compress_rate[index] = 0.7
                self.mask_index.append(index)
                # self.mask_index.append(index+1)
                # self.mask_index.append(index+2)

    def init_mask(self, layer_rate):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                if len(item.data.size())==4:
                    self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                                                               self.model_length[index])
                    self.mat[index] = self.convert2tensor(self.mat[index]).cuda()
                else:
                    self.mat[index] = self.mat[index - 1].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                if len(item.data.size()) == 4:
                    #print("do_mask , index: %d"%index)
                    for i in range(item.data.size()[0]):
                        item.data[i,:,:,:] = item.data[i,:,:,:] * self.mat[index][i]
                else:
                    for i in range(item.size()[0]):
                        item.data[i] = item.data[i] * self.mat[index][i]
                 #a = item.data.view(self.model_length[index])
                 #b = a * self.mat[index]
                 #item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if index in [x for x in range(0, 153, 3)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    m = Mask(model)
    m.init_length()
    m.model = model
    m.init_mask(1.0)
    # m.if_zero()
    m.do_mask()
    model = m.model.cuda()

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=opt.img_size,
        batch_size=8,
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    logger.list_of_scalars_summary(evaluation_metrics, 0)
    print(f"---- mAP {AP.mean()}")

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()
            # ----------------
            #   Log progress
            # ----------------
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
            # metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]
            #
            # # Log metrics at each YOLO layer
            # for i, metric in enumerate(metrics):
            #     formats = {m: "%.6f" for m in metrics}
            #     formats["grid_size"] = "%2d"
            #     formats["cls_acc"] = "%.2f%%"
            #     row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            #     metric_table += [[metric, *row_metrics]]
            #
            #     # Tensorboard logging
            #     tensorboard_log = []
            #     for j, yolo in enumerate(model.yolo_layers):
            #         for name, metric in yolo.metrics.items():
            #             if name != "grid_size":
            #                 tensorboard_log += [(f"{name}_{j+1}", metric)]
            #     tensorboard_log += [("loss", loss.item())]
            #     logger.list_of_scalars_summary(tensorboard_log, batches_done)
            #
            # log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"
            print(log_str)
            model.seen += imgs.size(0)

        m.model = model
        m.init_mask(1.0)
        m.if_zero()
        m.do_mask()
        m.if_zero()
        model = m.model.cuda()

        if epoch % opt.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            # print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
