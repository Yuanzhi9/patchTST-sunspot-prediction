"""CV5 trainer for PatchTST curve plus cycle-event auxiliary targets."""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from data_provider.cv5_event_data import build_cv5_event_data
from exp.exp_cv5 import Exp_CV5
from exp.exp_main import _load_checkpoint
from models import PatchTSTMultiTask
from utils.tools import EarlyStopping, adjust_learning_rate


MODE_TO_COLUMNS = {
    "none": [],
    "amp": [0],
    "peak_time": [1],
    "trough_time": [2],
    "all": [0, 1, 2],
}


class Exp_CV5_MultiTask(Exp_CV5):
    def _build_model(self):
        model = PatchTSTMultiTask.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if self._cv_bundle is None:
            self._cv_bundle = build_cv5_event_data(self.args)
        return self._cv_bundle[flag]

    def _curve_outputs(self, outputs):
        f_dim = -1 if self.args.features == "MS" else 0
        return outputs[:, -self.args.pred_len :, f_dim:]

    def _aux_loss_components(self, aux_pred, aux_true, aux_mask):
        columns = MODE_TO_COLUMNS[self.args.mt_mode]
        if not columns:
            return []
        losses = []
        for column in columns:
            mask = aux_mask[:, column].bool()
            if mask.any():
                losses.append(
                    F.smooth_l1_loss(
                        aux_pred[mask, column],
                        aux_true[mask, column],
                        beta=0.5,
                        reduction="mean",
                    )
                )
        return losses

    def _aux_loss(self, aux_pred, aux_true, aux_mask):
        losses = self._aux_loss_components(aux_pred, aux_true, aux_mask)
        if not losses:
            return aux_pred.sum() * 0.0
        if getattr(self.args, "mt_weighting", "fixed") == "uncertainty":
            model = (
                self.model.module
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model
            )
            log_vars = model.mt_log_vars
            if log_vars is None or len(log_vars) != len(losses):
                raise RuntimeError("uncertainty weighting requires matching mt_log_vars")
            return torch.stack(
                [
                    torch.exp(-log_vars[index]) * loss + log_vars[index]
                    for index, loss in enumerate(losses)
                ]
            ).mean()
        return torch.stack(losses).mean()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _, _, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, _ = self.model(batch_x)
                outputs = self._curve_outputs(outputs)
                f_dim = -1 if self.args.features == "MS" else 0
                targets = batch_y[:, -self.args.pred_len :, f_dim:]
                total_loss.append(float(criterion(outputs, targets).detach().cpu()))
        self.model.train()
        return float(np.mean(total_loss))

    def train(self, setting):
        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        if len(train_loader) == 0:
            raise ValueError("training loader has zero full batches")

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
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "train_total_loss",
                    "train_curve_loss",
                    "train_aux_loss",
                    "inner_val_curve_loss",
                    "seconds",
                    "best_val_curve_loss",
                ]
            )

        print(
            f"start multitask training: fold={self.args.cv_fold} "
            f"mode={self.args.mt_mode} train_windows={len(train_data)} "
            f"inner_val_windows={len(vali_data)}"
        )
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_time = time.time()
            totals, curves, auxiliaries = [], [], []
            for batch_x, batch_y, _, _, event_y, event_mask in train_loader:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                event_y = event_y.float().to(self.device)
                event_mask = event_mask.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, aux_pred = self.model(batch_x)
                        outputs = self._curve_outputs(outputs)
                        f_dim = -1 if self.args.features == "MS" else 0
                        targets = batch_y[:, -self.args.pred_len :, f_dim:]
                        curve_loss = criterion(outputs, targets)
                        aux_loss = self._aux_loss(aux_pred, event_y, event_mask)
                        loss = curve_loss + self.args.mt_aux_weight * aux_loss
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(model_optim)
                    amp_scaler.update()
                else:
                    outputs, aux_pred = self.model(batch_x)
                    outputs = self._curve_outputs(outputs)
                    f_dim = -1 if self.args.features == "MS" else 0
                    targets = batch_y[:, -self.args.pred_len :, f_dim:]
                    curve_loss = criterion(outputs, targets)
                    aux_loss = self._aux_loss(aux_pred, event_y, event_mask)
                    loss = curve_loss + self.args.mt_aux_weight * aux_loss
                    loss.backward()
                    model_optim.step()

                totals.append(float(loss.item()))
                curves.append(float(curve_loss.item()))
                auxiliaries.append(float(aux_loss.item()))
                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()

            train_total = float(np.mean(totals))
            train_curve = float(np.mean(curves))
            train_aux = float(np.mean(auxiliaries))
            vali_curve = self.vali(vali_data, vali_loader, criterion)
            elapsed = time.time() - epoch_time
            early_stopping(
                vali_curve,
                self.model,
                model_optim,
                scheduler,
                epoch,
                self.args,
                path,
            )
            best = float(early_stopping.val_loss_min)
            with open(history_path, "a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch + 1,
                        train_total,
                        train_curve,
                        train_aux,
                        vali_curve,
                        elapsed,
                        best,
                    ]
                )
            print(
                f"Epoch: {epoch + 1}, Steps: {len(train_loader)} | "
                f"Total: {train_total:.7f} Curve: {train_curve:.7f} "
                f"Aux: {train_aux:.7f} ValCurve: {vali_curve:.7f} "
                f"Best: {best:.7f} | {elapsed:.3f}s"
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != "TST":
                adjust_learning_rate(
                    model_optim, scheduler, epoch + 1, self.args
                )

        checkpoint_path = os.path.join(path, "full_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            raise RuntimeError("training finished without full_checkpoint.pth")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data("test")
        if test:
            checkpoint_path = os.path.join(
                self.args.checkpoints, setting, "full_checkpoint.pth"
            )
            checkpoint = _load_checkpoint(checkpoint_path)
            state = (
                checkpoint["model_state_dict"]
                if "model_state_dict" in checkpoint
                else checkpoint
            )
            self.model.load_state_dict(state)

        preds, trues, aux_preds, aux_trues, aux_masks = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, _, _, event_y, event_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, aux_pred = self.model(batch_x)
                outputs = self._curve_outputs(outputs)
                f_dim = -1 if self.args.features == "MS" else 0
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                aux_preds.append(aux_pred.detach().cpu().numpy())
                aux_trues.append(event_y.numpy())
                aux_masks.append(event_mask.numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        folder = os.path.join(self.args.results_root, setting)
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "pred.npy"), preds)
        np.save(os.path.join(folder, "true.npy"), trues)
        np.save(os.path.join(folder, "aux_pred.npy"), np.concatenate(aux_preds))
        np.save(os.path.join(folder, "aux_true.npy"), np.concatenate(aux_trues))
        np.save(os.path.join(folder, "aux_mask.npy"), np.concatenate(aux_masks))

        meta = test_data.metadata()
        train_data, _ = self._get_data("train")
        meta["n_train_windows"] = len(train_data)
        meta["train_window_sampling"] = train_data.metadata()["window_sampling"]
        meta["setting"] = setting
        meta["outer_evaluation_count"] = 1
        meta["mt_mode"] = self.args.mt_mode
        meta["mt_aux_weight"] = self.args.mt_aux_weight
        meta["mt_weighting"] = self.args.mt_weighting
        meta["aux_columns"] = [
            "next_peak_raw",
            "months_to_next_peak",
            "months_to_next_trough",
            "cycle_length",
        ]
        meta["aux_units"] = [
            "target standard deviations",
            "years",
            "years",
            "years",
        ]
        with open(os.path.join(folder, "cv_window_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        print(
            f"saved outer fold: pred={preds.shape} true={trues.shape} "
            f"target={folder}"
        )
