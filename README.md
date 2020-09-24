# Data preprocesing

```python
# load data as pandas dataframe
data = get_stallion_data()  
```

```python
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

```


```python
# Sample data preview
data.sample(10, random_state=521)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>agency</th>
      <th>sku</th>
      <th>volume</th>
      <th>date</th>
      <th>industry_volume</th>
      <th>soda_volume</th>
      <th>avg_max_temp</th>
      <th>price_regular</th>
      <th>price_actual</th>
      <th>discount</th>
      <th>...</th>
      <th>football_gold_cup</th>
      <th>beer_capital</th>
      <th>music_fest</th>
      <th>discount_in_percent</th>
      <th>timeseries</th>
      <th>time_idx</th>
      <th>month</th>
      <th>log_volume</th>
      <th>avg_volume_by_sku</th>
      <th>avg_volume_by_agency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>Agency_25</td>
      <td>SKU_03</td>
      <td>0.5076</td>
      <td>2013-01-01</td>
      <td>492612703</td>
      <td>718394219</td>
      <td>25.845238</td>
      <td>1264.162234</td>
      <td>1152.473405</td>
      <td>111.688829</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>8.835008</td>
      <td>228</td>
      <td>0</td>
      <td>1</td>
      <td>-0.678062</td>
      <td>1225.306376</td>
      <td>99.650400</td>
    </tr>
    <tr>
      <th>871</th>
      <td>Agency_29</td>
      <td>SKU_02</td>
      <td>8.7480</td>
      <td>2015-01-01</td>
      <td>498567142</td>
      <td>762225057</td>
      <td>27.584615</td>
      <td>1316.098485</td>
      <td>1296.804924</td>
      <td>19.293561</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1.465966</td>
      <td>177</td>
      <td>24</td>
      <td>1</td>
      <td>2.168825</td>
      <td>1634.434615</td>
      <td>11.397086</td>
    </tr>
    <tr>
      <th>19532</th>
      <td>Agency_47</td>
      <td>SKU_01</td>
      <td>4.9680</td>
      <td>2013-09-01</td>
      <td>454252482</td>
      <td>789624076</td>
      <td>30.665957</td>
      <td>1269.250000</td>
      <td>1266.490490</td>
      <td>2.759510</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.217413</td>
      <td>322</td>
      <td>8</td>
      <td>9</td>
      <td>1.603017</td>
      <td>2625.472644</td>
      <td>48.295650</td>
    </tr>
    <tr>
      <th>2089</th>
      <td>Agency_53</td>
      <td>SKU_07</td>
      <td>21.6825</td>
      <td>2013-10-01</td>
      <td>480693900</td>
      <td>791658684</td>
      <td>29.197727</td>
      <td>1193.842373</td>
      <td>1128.124395</td>
      <td>65.717978</td>
      <td>...</td>
      <td>-</td>
      <td>beer_capital</td>
      <td>-</td>
      <td>5.504745</td>
      <td>240</td>
      <td>9</td>
      <td>10</td>
      <td>3.076505</td>
      <td>38.529107</td>
      <td>2511.035175</td>
    </tr>
    <tr>
      <th>9755</th>
      <td>Agency_17</td>
      <td>SKU_02</td>
      <td>960.5520</td>
      <td>2015-03-01</td>
      <td>515468092</td>
      <td>871204688</td>
      <td>23.608120</td>
      <td>1338.334248</td>
      <td>1232.128069</td>
      <td>106.206179</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>music_fest</td>
      <td>7.935699</td>
      <td>259</td>
      <td>26</td>
      <td>3</td>
      <td>6.867508</td>
      <td>2143.677462</td>
      <td>396.022140</td>
    </tr>
    <tr>
      <th>7561</th>
      <td>Agency_05</td>
      <td>SKU_03</td>
      <td>1184.6535</td>
      <td>2014-02-01</td>
      <td>425528909</td>
      <td>734443953</td>
      <td>28.668254</td>
      <td>1369.556376</td>
      <td>1161.135214</td>
      <td>208.421162</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>15.218151</td>
      <td>21</td>
      <td>13</td>
      <td>2</td>
      <td>7.077206</td>
      <td>1566.643589</td>
      <td>1881.866367</td>
    </tr>
    <tr>
      <th>19204</th>
      <td>Agency_11</td>
      <td>SKU_05</td>
      <td>5.5593</td>
      <td>2017-08-01</td>
      <td>623319783</td>
      <td>1049868815</td>
      <td>31.915385</td>
      <td>1922.486644</td>
      <td>1651.307674</td>
      <td>271.178970</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>14.105636</td>
      <td>17</td>
      <td>55</td>
      <td>8</td>
      <td>1.715472</td>
      <td>1385.225478</td>
      <td>109.699200</td>
    </tr>
    <tr>
      <th>8781</th>
      <td>Agency_48</td>
      <td>SKU_04</td>
      <td>4275.1605</td>
      <td>2013-03-01</td>
      <td>509281531</td>
      <td>892192092</td>
      <td>26.767857</td>
      <td>1761.258209</td>
      <td>1546.059670</td>
      <td>215.198539</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>music_fest</td>
      <td>12.218455</td>
      <td>151</td>
      <td>2</td>
      <td>3</td>
      <td>8.360577</td>
      <td>1757.950603</td>
      <td>1925.272108</td>
    </tr>
    <tr>
      <th>2540</th>
      <td>Agency_07</td>
      <td>SKU_21</td>
      <td>0.0000</td>
      <td>2015-10-01</td>
      <td>544203593</td>
      <td>761469815</td>
      <td>28.987755</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.000000</td>
      <td>300</td>
      <td>33</td>
      <td>10</td>
      <td>-18.420681</td>
      <td>0.000000</td>
      <td>2418.719550</td>
    </tr>
    <tr>
      <th>12084</th>
      <td>Agency_21</td>
      <td>SKU_03</td>
      <td>46.3608</td>
      <td>2017-04-01</td>
      <td>589969396</td>
      <td>940912941</td>
      <td>32.478910</td>
      <td>1675.922116</td>
      <td>1413.571789</td>
      <td>262.350327</td>
      <td>...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>15.654088</td>
      <td>181</td>
      <td>51</td>
      <td>4</td>
      <td>3.836454</td>
      <td>2034.293024</td>
      <td>109.381800</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31 columns</p>
</div>




```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>volume</th>
      <th>industry_volume</th>
      <th>soda_volume</th>
      <th>avg_max_temp</th>
      <th>price_regular</th>
      <th>price_actual</th>
      <th>discount</th>
      <th>avg_population_2017</th>
      <th>avg_yearly_household_income_2017</th>
      <th>discount_in_percent</th>
      <th>timeseries</th>
      <th>time_idx</th>
      <th>log_volume</th>
      <th>avg_volume_by_sku</th>
      <th>avg_volume_by_agency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>21000.000000</td>
      <td>2.100000e+04</td>
      <td>2.100000e+04</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
      <td>2.100000e+04</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
      <td>21000.00000</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
      <td>21000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1492.403982</td>
      <td>5.439214e+08</td>
      <td>8.512000e+08</td>
      <td>28.612404</td>
      <td>1451.536344</td>
      <td>1267.347450</td>
      <td>184.374146</td>
      <td>1.045065e+06</td>
      <td>151073.494286</td>
      <td>10.574884</td>
      <td>174.50000</td>
      <td>29.500000</td>
      <td>2.464118</td>
      <td>1492.403982</td>
      <td>1492.403982</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2711.496882</td>
      <td>6.288022e+07</td>
      <td>7.824340e+07</td>
      <td>3.972833</td>
      <td>683.362417</td>
      <td>587.757323</td>
      <td>257.469968</td>
      <td>9.291926e+05</td>
      <td>50409.593114</td>
      <td>9.590813</td>
      <td>101.03829</td>
      <td>17.318515</td>
      <td>8.178218</td>
      <td>1051.790829</td>
      <td>1328.239698</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>4.130518e+08</td>
      <td>6.964015e+08</td>
      <td>16.731034</td>
      <td>0.000000</td>
      <td>-3121.690141</td>
      <td>0.000000</td>
      <td>1.227100e+04</td>
      <td>90240.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>-18.420681</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.272388</td>
      <td>5.090553e+08</td>
      <td>7.890880e+08</td>
      <td>25.374816</td>
      <td>1311.547158</td>
      <td>1178.365653</td>
      <td>54.935108</td>
      <td>6.018900e+04</td>
      <td>110057.000000</td>
      <td>3.749628</td>
      <td>87.00000</td>
      <td>14.750000</td>
      <td>2.112923</td>
      <td>932.285496</td>
      <td>113.420250</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>158.436000</td>
      <td>5.512000e+08</td>
      <td>8.649196e+08</td>
      <td>28.479272</td>
      <td>1495.174592</td>
      <td>1324.695705</td>
      <td>138.307225</td>
      <td>1.232242e+06</td>
      <td>131411.000000</td>
      <td>8.948990</td>
      <td>174.50000</td>
      <td>29.500000</td>
      <td>5.065351</td>
      <td>1402.305264</td>
      <td>1730.529771</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1774.793475</td>
      <td>5.893715e+08</td>
      <td>9.005551e+08</td>
      <td>31.568405</td>
      <td>1725.652080</td>
      <td>1517.311427</td>
      <td>272.298630</td>
      <td>1.729177e+06</td>
      <td>206553.000000</td>
      <td>15.647058</td>
      <td>262.00000</td>
      <td>44.250000</td>
      <td>7.481439</td>
      <td>2195.362302</td>
      <td>2595.316500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>22526.610000</td>
      <td>6.700157e+08</td>
      <td>1.049869e+09</td>
      <td>45.290476</td>
      <td>19166.625000</td>
      <td>4925.404000</td>
      <td>19166.625000</td>
      <td>3.137874e+06</td>
      <td>247220.000000</td>
      <td>226.740147</td>
      <td>349.00000</td>
      <td>59.000000</td>
      <td>10.022453</td>
      <td>4332.363750</td>
      <td>5884.717375</td>
    </tr>
  </tbody>
</table>
</div>



# Create dataset and dataloaders


```python
# use the last six months as a validation set, and compare to forcast result
max_prediction_length = 6  # forecast 6 months
max_encoder_length = 24  # use 24 months of history
training_cutoff = data["time_idx"].max() - max_prediction_length

# Normalize data: scale each time series separately and indicate that values are always positive
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
```


```python
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
```


```python
# create validation set (predict=True) which means to predict the
# last max_prediction_length points in time for each series
validation = TimeSeriesDataSet.from_dataset(
    training, data, predict=True, stop_randomization=True
)
```


```python
# create dataloaders for model
batch_size = 128
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)
```

# Create Baseline Model as benchmark


```python
import torch
from pytorch_forecasting import Baseline

# calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
actuals = torch.cat([y for x, y in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()

```




    293.0088195800781



# Find optimal learning rate


```python
import pytorch_lightning as pl
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
```


```python
pl.seed_everything(42)
trainer = pl.Trainer(
    gpus=0,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,# not meaningful for finding the learning rate but otherwise very important
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)

```

    GPU available: True, used: False
    TPU available: False, using: 0 TPU cores



```python
# Number of parameters in network
tft.size()
```




    29625




```python
res = trainer.lr_find(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, max_lr=10., min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
```


       | Name                               | Type                            | Params
    ----------------------------------------------------------------------------------------
    0  | loss                               | QuantileLoss                    | 0     
    1  | input_embeddings                   | ModuleDict                      | 1 K   
    2  | prescalers                         | ModuleDict                      | 256   
    3  | static_variable_selection          | VariableSelectionNetwork        | 3 K   
    4  | encoder_variable_selection         | VariableSelectionNetwork        | 8 K   
    5  | decoder_variable_selection         | VariableSelectionNetwork        | 2 K   
    6  | static_context_variable_selection  | GatedResidualNetwork            | 1 K   
    7  | static_context_initial_hidden_lstm | GatedResidualNetwork            | 1 K   
    8  | static_context_initial_cell_lstm   | GatedResidualNetwork            | 1 K   
    9  | static_context_enrichment          | GatedResidualNetwork            | 1 K   
    10 | lstm_encoder                       | LSTM                            | 2 K   
    11 | lstm_decoder                       | LSTM                            | 2 K   
    12 | post_lstm_gate_encoder             | GatedLinearUnit                 | 544   
    13 | post_lstm_add_norm_encoder         | AddNorm                         | 32    
    14 | static_enrichment                  | GatedResidualNetwork            | 1 K   
    15 | multihead_attn                     | InterpretableMultiHeadAttention | 1 K   
    16 | post_attn_gate_norm                | GateAddNorm                     | 576   
    17 | pos_wise_ff                        | GatedResidualNetwork            | 1 K   
    18 | pre_output_gate_norm               | GateAddNorm                     | 576   
    19 | output_layer                       | Linear                          | 119   



    HBox(children=(FloatProgress(value=0.0, description='Finding best initial lr', style=ProgressStyle(description…


    Saving latest checkpoint..
    LR finder stopped early due to diverging loss.


    suggested learning rate: 0.15135612484362077



![png](./readme_img/output_17_4.png)


#### Optimal learning is lower than suggested learning rate, so learning rate used will be mark down a little bit, 0.03

# Training the Temporal Fusion Transformer with PyTorch Lightning


```python
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
```


```python
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
logger = TensorBoardLogger("lightning_logs")  # log result to tensorboard

# create trainer using PyTorch Lightning
trainer = pl.Trainer(
    max_epochs=30,
    gpus=[0],  # 0 to train on CPU whereas [0] for GPU
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
```

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    CUDA_VISIBLE_DEVICES: [0]



```python
# Number of parameters in network
tft.size()
```




    29625




```python
# fit network
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)
```


       | Name                               | Type                            | Params
    ----------------------------------------------------------------------------------------
    0  | loss                               | QuantileLoss                    | 0     
    1  | input_embeddings                   | ModuleDict                      | 1 K   
    2  | prescalers                         | ModuleDict                      | 256   
    3  | static_variable_selection          | VariableSelectionNetwork        | 3 K   
    4  | encoder_variable_selection         | VariableSelectionNetwork        | 8 K   
    5  | decoder_variable_selection         | VariableSelectionNetwork        | 2 K   
    6  | static_context_variable_selection  | GatedResidualNetwork            | 1 K   
    7  | static_context_initial_hidden_lstm | GatedResidualNetwork            | 1 K   
    8  | static_context_initial_cell_lstm   | GatedResidualNetwork            | 1 K   
    9  | static_context_enrichment          | GatedResidualNetwork            | 1 K   
    10 | lstm_encoder                       | LSTM                            | 2 K   
    11 | lstm_decoder                       | LSTM                            | 2 K   
    12 | post_lstm_gate_encoder             | GatedLinearUnit                 | 544   
    13 | post_lstm_add_norm_encoder         | AddNorm                         | 32    
    14 | static_enrichment                  | GatedResidualNetwork            | 1 K   
    15 | multihead_attn                     | InterpretableMultiHeadAttention | 1 K   
    16 | post_attn_gate_norm                | GateAddNorm                     | 576   
    17 | pos_wise_ff                        | GatedResidualNetwork            | 1 K   
    18 | pre_output_gate_norm               | GateAddNorm                     | 576   
    19 | output_layer                       | Linear                          | 119   



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validation sanity check', layout=Layout…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Training', layout=Layout(flex='2'), max…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Validating', layout=Layout(flex='2'), m…


    Saving latest checkpoint..








    1



## Training data in Tensorboard (predictions on the training and validation set)


```python

```

## Evaluate the trained model


```python
from pytorch_forecasting.metrics import MAE

# load the best model according to the validation loss (given that
# we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
```


```python
# calculate mean absolute error on validation set
actuals = torch.cat([y for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()
```




    tensor(249.1484)




```python
MAE(predictions, actuals)
```




    MAE()




```python
# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
```


```python
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);
```


![png](./readme_img/output_31_0.png)



![png](./readme_img/output_31_1.png)



![png](./readme_img/output_31_2.png)



![png](./readme_img/output_31_3.png)



![png](./readme_img/output_31_4.png)



![png](./readme_img/output_31_5.png)



![png](./readme_img/output_31_6.png)



![png](./readme_img/output_31_7.png)



![png](./readme_img/output_31_8.png)



![png](./readme_img/output_31_9.png)


## Worst Performers

#### Looking at the worst performers in terms of sMAPE gives us an idea where the model has issues with forecasting reliably


```python
from pytorch_forecasting.metrics import SMAPE

# calcualte metric by which to display
predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses

# show 10 examples for demonstration purposes
for idx in range(10): # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE());
```


![png](./readme_img/output_34_0.png)



![png](./readme_img/output_34_1.png)



![png](./readme_img/output_34_2.png)



![png](./readme_img/output_34_3.png)



![png](./readme_img/output_34_4.png)



![png](./readme_img/output_34_5.png)



![png](./readme_img/output_34_6.png)



![png](./readme_img/output_34_7.png)



![png](./readme_img/output_34_8.png)



![png](./readme_img/output_34_9.png)


## actuals vs predictions


```python
predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictioans_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);
```


![png](./readme_img/output_36_0.png)



![png](./readme_img/output_36_1.png)



![png](./readme_img/output_36_2.png)



![png](./readme_img/output_36_3.png)



![png](./readme_img/output_36_4.png)



![png](./readme_img/output_36_5.png)



![png](./readme_img/output_36_6.png)



![png](./readme_img/output_36_7.png)



![png](./readme_img/output_36_8.png)



![png](./readme_img/output_36_9.png)



![png](./readme_img/output_36_10.png)



![png](./readme_img/output_36_11.png)



![png](./readme_img/output_36_12.png)



![png](./readme_img/output_36_13.png)



![png](./readme_img/output_36_14.png)



![png](./readme_img/output_36_15.png)



![png](./readme_img/output_36_16.png)



![png](./readme_img/output_36_17.png)



![png](./readme_img/output_36_18.png)



![png](./readme_img/output_36_19.png)


## Interpret model


```python
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)
```




    {'attention': <Figure size 432x288 with 1 Axes>,
     'static_variables': <Figure size 504x270 with 1 Axes>,
     'encoder_variables': <Figure size 504x378 with 1 Axes>,
     'decoder_variables': <Figure size 504x252 with 1 Axes>}




![png](./readme_img/output_38_1.png)



![png](./readme_img/output_38_2.png)



![png](./readme_img/output_38_3.png)



![png](./readme_img/output_38_4.png)


#### As observered, price related variables are the among the top 2 predictors for both encoder and decoder. Next, past observed volume is statistically proven to be the most important static and encoder variable.  Time related variables seem to rather lless important, this may prove that recent data are more significant than the older ones.

## Partial dependency


```python
dependency = best_tft.predict_dependency(val_dataloader.dataset, "discount_in_percent", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe")b
```


    HBox(children=(FloatProgress(value=0.0, description='Predict', max=30.0, style=ProgressStyle(description_width…



```python
# plotting median and 25% and 75% percentile
agg_dependency = dependency.groupby("discount_in_percent").normalized_prediction.agg(median="median", q25=lambda x: x.quantile(.25),  q75=lambda x: x.quantile(.75))
ax = agg_dependency.plot(y="median")
ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=.3);

```


![png](./readme_img/output_42_0.png)


####  Interpret model better (assume independence of features)
