"""Independent CV5 experiment class for BASE-2026-11-M.

New file only. It subclasses the legacy Exp_Main for model construction and
metric helpers, but replaces data routing and the training loop.
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler

from data_provider.cv5_data import build_cv5_data
from exp.exp_main import Exp_Main, _load_checkpoint
from utils.tools import EarlyStopping, adjust_learning_rate


class Exp_CV5(Exp_Main):
    def __init__(self, args):
        super().__init__(args)
        self._cv_bundle = None

    def _get_data(self, flag):
        if self._cv_bundle is None:
            self._cv_bundle = build_cv5_data(self.args)
        return self._cv_bundle[flag]

    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        amp_scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        history_path = os.path.join(path, "train_history.csv")
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "inner_val_loss", "seconds", "best_val_loss"])

        print(
            f"start training: fold={self.args.cv_fold} "
            f"train_windows={len(train_data)} inner_val_windows={len(vali_data)}"
        )
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            train_losses = []

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        targets = batch_y[:, -self.args.pred_len :, f_dim:]
                        loss = criterion(outputs, targets)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(model_optim)
                    amp_scaler.update()
                else:
                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    targets = batch_y[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(outputs, targets)
                    loss.backward()
                    model_optim.step()

                train_losses.append(float(loss.item()))
                if self.args.lradj == "TST":
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            train_loss = float(np.mean(train_losses))
            vali_loss = float(self.vali(vali_data, vali_loader, criterion))
            elapsed = time.time() - epoch_time
            early_stopping(vali_loss, self.model, model_optim, scheduler, epoch, self.args, path)
            best = float(early_stopping.val_loss_min)
            with open(history_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([epoch + 1, train_loss, vali_loss, elapsed, best])

            print(
                f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | "
                f"Train Loss: {train_loss:.7f} Inner Val Loss: {vali_loss:.7f} "
                f"Best: {best:.7f} | {elapsed:.3f}s"
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f"Updating learning rate to {scheduler.get_last_lr()[0]}")

        if not os.path.exists(os.path.join(path, "full_checkpoint.pth")):
            raise RuntimeError("training finished without full_checkpoint.pth")
        return self.model

    def test(self, setting, test=0):
        test_data, _ = self._get_data("test")
        if test:
            checkpoint_path = os.path.join(self.args.checkpoints, setting, "full_checkpoint.pth")
            checkpoint = _load_checkpoint(checkpoint_path)
            state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state)

        preds = []
        trues = []
        self.model.eval()
        test_loader = self._cv_bundle["test"][1]
        with torch.no_grad():
            for batch_x, batch_y, _, _ in test_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        folder = os.path.join(self.args.results_root, setting)
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "pred.npy"), preds)
        np.save(os.path.join(folder, "true.npy"), trues)

        meta = test_data.metadata()
        train_data, _ = self._get_data("train")
        meta["n_train_windows"] = len(train_data)
        meta["train_window_sampling"] = train_data.metadata()["window_sampling"]
        meta["setting"] = setting
        meta["outer_evaluation_count"] = 1
        with open(os.path.join(folder, "cv_window_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(
            f"saved outer fold: pred={preds.shape} true={trues.shape} "
            f"target={folder}"
        )
