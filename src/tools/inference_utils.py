import os
from os import path
from typing import Optional, Tuple, Union, List

import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


#modules
from ffm import FFM, FFmHparams

from ffm.pytorch_patched_decoder_MOE import PatchedTimeSeriesDecoder_MOE

from tools.utils import log_model_statistics, make_logging_file 
from tools.model_utils import get_model_FFM





class TimeSeriesDataset_SingleCSV_Inference(Dataset):
    """
    Inference-only dataset for a SINGLE CSV.

    Output signature matches your training dataset:
        (x_context, x_padding, freq, x_future)

    Shapes/dtypes:
        x_context: [L, 1], float32
        x_padding: [L, 1], float32 (zeros; no masking at inference)
        freq:      [1],   int64
        x_future:  [0, 1], float32 (empty; no ground truth at inference)

    Modes:
      - sliding_windows=False: one sample per selected column (the LAST L values).
      - sliding_windows=True: stride=1 over the series, generating all full windows.

    Notes:
      - Each selected column is treated as an independent univariate series.
      - If you normalized during training, prefer doing the SAME normalization upstream.
        `series_norm=True` here applies simple per-series z-score on the full column.
    """

    def __init__(
        self,
        csv_path: str,
        context_length: int,
        freq_type: int,
        columns: Optional[List[Union[int, str]]] = None,
        first_c_date: bool = True,
        series_norm: bool = False,
        dropna: bool = True,
        sliding_windows: bool = False,   # NEW: stride=1 windows if True
    ):
        super().__init__()
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        self.csv_path = csv_path
        self.L = int(context_length)
        self.freq_type = int(freq_type)
        self.first_c_date = bool(first_c_date)
        self.series_norm = bool(series_norm)
        self.dropna = bool(dropna)
        self.sliding_windows = bool(sliding_windows)

        # ---- Load CSV
        df = pd.read_csv(csv_path)

        # ---- Resolve columns to use
        if columns is None:
            start_idx = 1 if self.first_c_date else 0
            use_cols = list(df.columns[start_idx:])
        else:
            use_cols = []
            ncols = len(df.columns)
            for c in columns:
                if isinstance(c, int):
                    if c < 0 or c >= ncols:
                        raise IndexError(
                            f"Column index {c} out of range [0, {ncols-1}] for CSV '{csv_path}'."
                        )
                    use_cols.append(df.columns[c])
                elif isinstance(c, str):
                    if c not in df.columns:
                        raise KeyError(f"Column '{c}' not found in CSV '{csv_path}'.")
                    use_cols.append(c)
                else:
                    raise TypeError("columns entries must be int indices or str names.")

        # ---- Optional row-wise NaN drop (before numeric coercion)
        if self.dropna:
            df = df.dropna(axis=0).reset_index(drop=True)

        # ---- Build per-series numeric arrays
        self.series_arrays: List[np.ndarray] = []
        for c in use_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            arr = s.to_numpy(dtype=np.float32)
            if self.dropna:
                arr = arr[~np.isnan(arr)]
            else:
                # if not dropping, still ensure enough valid tail for slicing
                if np.isnan(arr).any():
                    raise ValueError(
                        f"Column '{c}' contains NaNs; set dropna=True or clean prior to loading."
                    )

            if len(arr) < self.L:
                raise ValueError(
                    f"Column '{c}' too short: len={len(arr)} < context_length={self.L}."
                )

            if self.series_norm:
                mu = float(arr.mean())
                sigma = float(arr.std(ddof=0))
                if sigma == 0.0:
                    sigma = 1.0
                arr = (arr - mu) / sigma

            self.series_arrays.append(arr)

        # ---- Build indices for __getitem__
        # For compatibility with your length-aware sampler, we maintain sample_lengths.
        self.index_records: List[Tuple[int, int]] = []  # (series_idx, start_idx) only used in sliding mode
        if self.sliding_windows:
            # stride=1 windows for each series
            for sidx, arr in enumerate(self.series_arrays):
                n = len(arr)
                # number of full windows: n - L + 1
                for start in range(0, n - self.L + 1):
                    self.index_records.append((sidx, start))
            self.sample_lengths = [self.L] * len(self.index_records)
        else:
            # one sample per series (the last window)
            self.sample_lengths = [self.L] * len(self.series_arrays)

    def __len__(self) -> int:
        if self.sliding_windows:
            return len(self.index_records)
        return len(self.series_arrays)

    def get_length(self, idx: int) -> int:
        return self.sample_lengths[idx]

    def __getitem__(self, idx: int):
        if self.sliding_windows:
            series_idx, start_idx = self.index_records[idx]
            series = self.series_arrays[series_idx]
            ctx_np = series[start_idx : start_idx + self.L]
        else:
            series = self.series_arrays[idx]
            ctx_np = series[-self.L:]  # last L points

        # Tensors
        x_context = torch.as_tensor(ctx_np, dtype=torch.float32).unsqueeze(-1)  # [L,1]
        x_padding = torch.zeros((self.L, 1), dtype=torch.float32)               # [L,1]
        freq = torch.tensor([self.freq_type], dtype=torch.long)                 # [1]
        x_future = torch.empty((0, 1), dtype=torch.float32)                     # [0,1]

        return x_context, x_padding, freq, x_future

    # Convenience: get a list of all samples (useful without a DataLoader)
    def as_list(self):
        return [self[i] for i in range(len(self))]

    def __repr__(self) -> str:
        mode = "sliding" if self.sliding_windows else "last-only"
        return (f"{self.__class__.__name__}(csv_path='{self.csv_path}', "
                f"L={self.L}, freq={self.freq_type}, num_series={len(self.series_arrays)}, "
                f"mode={mode}, series_norm={self.series_norm}, dropna={self.dropna})")








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