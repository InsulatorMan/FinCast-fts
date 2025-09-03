import os
from os import path
from typing import List, Optional, Tuple, Union, Dict, Any
from types import SimpleNamespace

import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


#modules
from ffm import FFM, FFmHparams

from ffm.pytorch_patched_decoder_MOE import PatchedTimeSeriesDecoder_MOE

from tools.utils import log_model_statistics, make_logging_file 
from tools.model_utils import get_model_FFM

from data_tools.Inference_dataset import *
from contextlib import nullcontext






















class FinCast_Inference:


    def __init__(
            self,
            config: SimpleNamespace,
            ):
        
        self.config = config

        self.inference_freq = freq_reader_inference(config.data_frequency)

        self.inference_dataset = TimeSeriesDataset_SingleCSV_Inference(csv_path=self.config.data_path,
                                                                       context_length=self.config.context_len,
                                                                       freq_type=self.config.inference_freq,
                                                                       columns=self.config.columns_target,
                                                                       first_c_date=True,
                                                                       series_norm=self.config.series_norm,
                                                                       dropna=False,
                                                                       sliding_windows=self.config.all_data)
        



        if self.config.model_version.lower() == "v1":
            self.config.num_experts = 4
            self.config.gating_top_n = 2
            self.config.load_from_compile = True

        self.model_api = get_model_api(self.config)


    
    # -------- Inference runner with mapping (tensor forward) --------
    def run_inference_with_mapping(
        model,
        ds: TimeSeriesDataset_SingleCSV_Inference,
        batch_size: int = 256,
        num_workers: int = 4,
        device: Optional[str] = "cuda",
        amp: bool = True,
    ):
        """
        Runs inference (sliding or last-window) and returns:
        preds_all: torch.Tensor on CPU, batch-dim first
        mapping_df: pandas.DataFrame with one row per sample (requires ds.return_meta=True)

        Notes:
        - Uses inference_mode for speed.
        - Uses AMP only on CUDA; otherwise falls back to a no-op context.
        """
        if not ds.return_meta:
            raise ValueError("Construct the dataset with return_meta=True to obtain mapping.")

        loader = make_inference_loader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        model.eval()
        preds_collector: List[torch.Tensor] = []
        mapping_rows: List[Dict[str, Any]] = []

        use_cuda = (device is not None and "cuda" in device and torch.cuda.is_available())
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (amp and use_cuda) else nullcontext()

        with torch.inference_mode(), amp_ctx:
            for x_ctx, x_pad, freq, _x_fut, meta in loader:
                if device is not None:
                    x_ctx = x_ctx.to(device, non_blocking=True)
                    freq  = freq.to(device, non_blocking=True)

                # Forward (adapt call signature to your model)
                preds = model(x_ctx, freq=freq)
                preds_collector.append(preds.detach().cpu())

                # Meta aligns with batch samples
                mapping_rows.extend(meta)

        preds_all = torch.cat(preds_collector, dim=0) if preds_collector else torch.empty(0)
        mapping_df = pd.DataFrame(mapping_rows) if mapping_rows else None
        return preds_all, mapping_df











def collate_with_optional_meta(batch: List[Tuple]):
    """
    Works for both sample formats:
      - (x_ctx, x_pad, freq, x_fut)
      - (x_ctx, x_pad, freq, x_fut, meta)

    Returns:
      x_ctx: [B,L,1]
      x_pad: [B,L,1]
      freq : [B,1]
      x_fut: [B,0,1]
      meta : list[dict] | None
    """
    has_meta = (len(batch[0]) == 5)

    x_ctx  = torch.stack([b[0] for b in batch], dim=0)
    x_pad  = torch.stack([b[1] for b in batch], dim=0)
    freq   = torch.stack([b[2] for b in batch], dim=0)
    x_fut  = torch.stack([b[3] for b in batch], dim=0)
    meta   = [b[4] for b in batch] if has_meta else None

    return x_ctx, x_pad, freq, x_fut, meta































def freq_reader_inference(freq: str) -> int:

    freq = str.upper(freq)
    if freq.endswith("MS"):
        return 1
    elif freq.endswith(("H", "T", "MIN", "D", "B", "U", "S")):
        return 0
    elif freq.endswith(("W", "M")):
        return 1
    elif freq.endswith(("Y", "Q", "A")):
        return 2
    else:
        print("Invalid frequency, default to fast freq: 0")
        return 0






def get_forecasts_f(model, past, freq):
  """Get forecasts. add for median and quantiile output supports"""
 
  
  lfreq = [freq] * past.shape[0]
  out, full_output = model.forecast(list(past), lfreq)


  return out, full_output





def get_model_api(config):
  model_path = config.model_path

  ffm_hparams = FFmHparams(
      backend="gpu",
      per_core_batch_size=32,
      horizon_len=config.horizon_len,  #variable
      context_len=config.context_len,  # Context length can be anything up to 2048 in multiples of 32
      use_positional_embedding=False,
      num_experts=config.num_experts,
      gating_top_n=config.gating_top_n,
      load_from_compile=config.load_from_compile,
      point_forecast_mode=config.forecast_mode,
  )


  model_actual, ffm_config, ffm_api = get_model_FFM(model_path, ffm_hparams)

  ffm_api.model_eval_mode()

  log_model_statistics(model_actual)

  return ffm_api




























## legacy code



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