# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:02:52 2020

@author: ylim
"""
"""
Data Preprocessing
"""
# Import dependencies
from pytorch_forecasting.data.examples import get_stallion_data
import numpy as np

# load data as pandas dataframe
data = get_stallion_data()  

#%%
# Make sur each row can be identified with a time step and a time series.
# add time index that is incremented by one for each time step.
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# Add additional features
# categories have to be strings
data["month"] = data["date"].dt.month.astype(str).astype("category")
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = (data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean"))
data["avg_volume_by_agency"] = (data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean"))

# Encode special days as unique identifier 
# first reverse one-hot encoding
special_days = [
    "easter_day", "good_friday", "new_year", "christmas",
    "labor_day", "independence_day", "revolution_day_memorial",
    "regional_games", "fifa_u_17_world_cup", "football_gold_cup",
    "beer_capital", "music_fest"
]
data[special_days] = (
    data[special_days]
    .apply(lambda x: x.map({0: "-", 1: x.name}))
    .astype("category")
)

# Sample data preview
data.sample(10, random_state=521)

#%%
data.describe()

#%%
# use the last six months as a validation set, and compare to forcast result
max_prediction_length = 6  # forecast 6 months
max_encoder_length = 24  # use 24 months of history
training_cutoff = data["time_idx"].max() - max_prediction_length

# Normalize data: scale each time series separately and indicate that values are always positive
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer

# Create training set
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=0,  # allow predictions without history
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=[
        "avg_population_2017",
        "avg_yearly_household_income_2017"
    ],
    time_varying_known_categoricals=["special_days", "month"],
    # group of categorical variables can be treated as 
    # one variable --> special days' list
    variable_groups={"special_days": special_days},
    time_varying_known_reals=[
        "time_idx",
        "price_regular",
        "discount_in_percent"
    ],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], coerce_positive=1.0
    ),  # use softplus with beta=1.0 and normalize by group
    add_relative_time_idx=True,  # add as feature
    add_target_scales=True,  # add as feature
    add_encoder_length=True,  # add as feature
)

# create validation set (predict=True) which means to predict the
# last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)
# create dataloaders for model
batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

#%%
"""
Training the Temporal Fusion Transformer with PyTorch Lightning
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
     
# Halt training when loss metric does not improve on validation set
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode="min"
)

#Log data
lr_logger = LearningRateLogger()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # log to tensorboard

# create trainer using PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=30,
    gpus=0,  # train on CPU, use gpus = [0] to run on GPU
    gradient_clip_val=0.1,
    early_stop_callback=early_stop_callback,
    limit_train_batches=30,  # running validation every 30 batches
    # fast_dev_run=True,  # comment in to quickly check for bugs
    callbacks=[lr_logger],
    logger=logger,
)
# initialise model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # biggest influence network size
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # QuantileLoss has 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # log example every 10 batches
    reduce_on_plateau_patience=4,  # reduce learning automatically
)
tft.size() # 29.6k parameters in model
# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)

#%%
"""
Evaluating the trained model
"""

from pytorch_forecasting.metrics import MAE
import torch

# load the best model according to the validation loss (given that
# we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# calculate mean absolute error on validation set
actuals = torch.cat([y for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
MAE(predictions, actuals)

from pytorch_forecasting.metrics import SMAPE
raw_predictions = best_tft.predict(val_dataloader, mode="raw")


# calculate metric by which to display
predictions, x = best_tft.predict(val_dataloader, return_x=True)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses
# show only two examples for demonstration purposes
for idx in range(2):
    best_tft.plot_prediction(
        x,
        raw_predictions,
        idx=indices[idx],
        add_loss_to_title=SMAPE()
    )
    
interpretation = best_tft.interpret_output(
    raw_predictions, reduction="sum"
)
best_tft.plot_interpretation(interpretation)