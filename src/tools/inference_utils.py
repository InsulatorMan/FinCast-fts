import os
from os import path
from typing import Optional, Tuple, Union, List

import logging

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import yfinance as yf # type: ignore
from absl import app, flags # type: ignore
from huggingface_hub import snapshot_download # type: ignore
from torch.utils.data import Dataset

from finetuning.finetuning_torch_MOE import FinetuningConfig, TimesFMFinetuner_ffm

#modules
from timesfm import TSFM, TimesFmCheckpoint,  TimesFmHparams
from ffm import FFM, FFmHparams

from ffm.pytorch_patched_decoder_MOE import PatchedTimeSeriesDecoder_MOE






def plot_predictions_multi(
    model: FFM,
    val_dataset: Dataset,
    number_img: int = 5,
    model_version: str = 'exp',
    save_dir: Optional[str] = None,
    save_name: Optional[str] = "predictions",
    moe_n: int = 0,
    moe_tk: int = 0,
    quantile: int = 0,
    output_length: int = 128,
) -> None:
    """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
        quantile: 0 is the mean, 1 to 9 for 9 quantile.
        model: Trained TimesFM model
        val_dataset: Validation dataset
        save_path: Path to save the plot
    """

    model.eval()
    device = next(model.parameters()).device

    for plt_i in range(number_img):
        x_context, x_padding, freq, x_future = val_dataset[plt_i]
        x_context = x_context.unsqueeze(0)  # Add batch dimension
        x_padding = x_padding.unsqueeze(0)
        freq = freq.unsqueeze(0)
        x_future = x_future.unsqueeze(0)

        x_context = x_context.to(device)
        x_padding = x_padding.to(device)
        freq = freq.to(device)
        x_future = x_future.to(device)

        with torch.no_grad():
            predictions, total_aux_loss = model(x_context, x_padding.float(), freq)
            predictions_mean = predictions[..., quantile]  # [B, N, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

        context_vals = x_context[0].cpu().numpy()
        future_vals = x_future[0, 0:output_length].cpu().numpy()
        pred_vals = last_patch_pred[0, 0:output_length].cpu().numpy()

        context_len = len(context_vals)
        horizon_len = len(future_vals)

        plt.figure(figsize=(12, 6))

        if quantile == 0:
            quantile_str = "mean"
        else:
            quantile_str = quantile

        plt.plot(range(context_len),
                context_vals,
                label="Historical Data",
                color="blue",
                linewidth=2)

        plt.plot(
            range(context_len, context_len + horizon_len),
            future_vals,
            label="Ground Truth",
            color="green",
            linestyle="--",
            linewidth=2,
        )

        plt.plot(range(context_len, context_len + horizon_len),
                pred_vals,
                label="Prediction",
                color="red",
                linewidth=2)

        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title("FFM_{}_M{}_T{} Predictions vs Ground Truth\ndistribution on {}".format(model_version, moe_n, moe_tk, quantile_str))
        plt.legend()
        plt.grid(True)

        plt.show()

        if save_dir != None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, '{}_{}.jpg'.format(save_name, plt_i))
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.close()



def plot_predictions_multi_distribution(
    model: FFM,
    val_dataset: Dataset,
    number_img: int = 5,
    model_version: str = 'exp',
    save_dir: Optional[str] = None,
    save_name: Optional[str] = "predictions",
    moe_n: int = 0,
    moe_tk: int = 0,
    quantiles: Optional[List[int]] = None,  # Now optional list
    output_length: int = 128,
    output_number: bool = False,
) -> None:
    """
    Plot model predictions (mean + optional quantiles) against ground truth.

    Args:
        model: Trained FFM model
        val_dataset: Validation dataset
        number_img: Number of plots to generate
        model_version: For title metadata
        save_dir: If specified, saves plots
        save_name: Base name for saved files
        moe_n, moe_tk: MoE config for title
        quantiles: Optional list of quantile indices (1–9). Mean (index 0) is always plotted.
        output_length: Forecast horizon length
    """
    model.eval()
    device = next(model.parameters()).device

    if quantiles is None:
        quantiles = []

    for plt_i in range(number_img):
        x_context, x_padding, freq, x_future = val_dataset[plt_i]
        x_context = x_context.unsqueeze(0).to(device)
        x_padding = x_padding.unsqueeze(0).to(device)
        freq = freq.unsqueeze(0).to(device)
        x_future = x_future.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions, _ = model(x_context, x_padding.float(), freq)
            # predictions: [B, N, horizon_len, Q]
            last_step_preds = predictions[:, -1, :, :]  # [B, horizon_len, Q]

            # Extract mean forecast (index 0)
            mean_pred = last_step_preds[0, :output_length, 0].cpu().numpy()

            context_vals = x_context[0].cpu().numpy()
            future_vals = x_future[0, :output_length].cpu().numpy()
            context_len = len(context_vals)
            horizon_len = len(future_vals)

            if output_number:
                print('output values')
                print(context_vals)
                print(future_vals)

            plt.figure(figsize=(12, 6))

            # Historical input
            plt.plot(range(context_len), context_vals, label="Historical", color="blue", linewidth=2)

            # Ground truth
            plt.plot(
                range(context_len, context_len + horizon_len),
                future_vals,
                label="Ground Truth",
                color="green",
                linestyle="--",
                linewidth=2,
            )

            # Mean forecast
            plt.plot(
                range(context_len, context_len + horizon_len),
                mean_pred,
                label="Prediction (mean)",
                color="red",
                linewidth=2,
            )

            # Optional quantile forecasts
            for q in quantiles:
                if q <= 0:
                    continue  # Skip mean index if redundantly included
                quantile_pred = last_step_preds[0, :output_length, q].cpu().numpy()
                plt.plot(
                    range(context_len, context_len + horizon_len),
                    quantile_pred,
                    label=f"Prediction (q{q})",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.title(f"FFM_{model_version}_M{moe_n}_T{moe_tk} Forecast\nQuantiles={quantiles}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{save_name}_{plt_i}.jpg')
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")

            plt.show()
            plt.close()



def plot_predictions_multi_distribution_v2(
    model: FFM,
    val_dataset: Dataset,
    number_img: int = 5,
    model_version: str = 'exp',
    save_dir: Optional[str] = None,
    save_name: Optional[str] = "predictions",
    moe_n: int = 0,
    moe_tk: int = 0,
    quantiles: Optional[List[int]] = None,  # Now optional list
    output_length: int = 128,
    output_number: bool = False,
) -> None:
    """
    Plot model predictions (mean + optional quantiles) against ground truth.

    Args:
        model: Trained FFM model
        val_dataset: Validation dataset
        number_img: Number of plots to generate
        model_version: For title metadata
        save_dir: If specified, saves plots
        save_name: Base name for saved files
        moe_n, moe_tk: MoE config for title
        quantiles: Optional list of quantile indices (1–9). Mean (index 0) is always plotted.
        output_length: Forecast horizon length
    """
    model.eval()
    device = next(model.parameters()).device

    if quantiles is None:
        quantiles = []

    for plt_i in range(number_img):
        x_context, x_padding, freq, x_future = val_dataset[plt_i]
        x_context = x_context.unsqueeze(0).to(device)
        x_padding = x_padding.unsqueeze(0).to(device)
        freq = freq.unsqueeze(0).to(device)
        x_future = x_future.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions, _ = model(x_context, x_padding.float(), freq)
            # predictions: [B, N, horizon_len, Q]
            last_step_preds = predictions[:, -1, :, :]  # [B, horizon_len, Q]

            # Extract mean forecast (index 0)
            mean_pred = last_step_preds[0, :output_length, 0].cpu().numpy()

            context_vals = x_context[0].cpu().numpy()
            future_vals = x_future[0, :output_length].cpu().numpy()
            context_len = len(context_vals)
            horizon_len = len(future_vals)

            if output_number:
                print('output values')
                print(context_vals)
                print(future_vals)

            plt.figure(figsize=(12, 6))

            # Concatenated ground truth (historical + future)
            gt_full = np.concatenate([context_vals, future_vals], axis=0)
            plt.plot(
                range(context_len + horizon_len),
                gt_full,
                label="Ground Truth",
                color="blue",
                linewidth=2,
            )


            # Mean forecast
            plt.plot(
                range(context_len, context_len + horizon_len),
                mean_pred,
                label="Forecast (point)",
                color="red",
                linewidth=3,
            )

            # Optional quantile forecasts
            for q in quantiles:
                if q <= 0:
                    continue  # Skip mean index if redundantly included
                quantile_pred = last_step_preds[0, :output_length, q].cpu().numpy()
                plt.plot(
                    range(context_len, context_len + horizon_len),
                    quantile_pred,
                    label=f"Quantile (Q{q})",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.8,
                )

            #plt.xlabel("Time Step")
            #plt.ylabel("Value")
            #plt.title(f"FFM_{model_version}_M{moe_n}_T{moe_tk} Forecast\nQuantiles={quantiles}")
            plt.legend(fontsize=15)
            plt.grid(True)
            plt.tight_layout()

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{save_name}_{plt_i}.jpg')
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")

            plt.show()
            plt.close()