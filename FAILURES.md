# FAILURES.md

## What didn't work and why

### 1. Learning rate too high (0.001)
First model trained with lr=0.001. Caused a spike in validation 
loss at epoch 2 (val_loss jumped to 2.35). Model was taking steps 
too large and overshooting. Fixed by reducing to 0.0005.

### 2. Overfitting with single dropout layer
First model had one Dropout(0.4) layer. Training accuracy hit 98.8% 
while val accuracy plateaued at 91.7% — 7% gap. Added second dropout 
layer and L2 regularization on Dense layer. Gap remained similar but 
val accuracy itself stayed stable instead of degrading.

### 3. EarlyStopping too aggressive
patience=3 stopped training at epoch 3 when using lower learning rate. 
Model hadn't warmed up yet. Increased patience to 5 — model then 
trained properly to epoch 16.

### 4. Conv1D destroys mask information
TextVectorization uses masking for padding zeros. Conv1D doesn't 
support masks — it processes padding as real data. Known Keras 
limitation. Acceptable tradeoff for the performance gain Conv1D gives.

### 5. Noisy labels in dataset
Label 1 (Positive) contains politically ambiguous tweets that are 
not clearly positive. Example: "isko zammen chayiy bus dogma hai..." 
labelled as Positive. Model learned this noise, causing misclassification 
of clearly negative product/service reviews as Positive.

Fix attempted: None — this is a data quality issue, not a model issue.
Real fix would require re-labelling the dataset with domain-specific 
annotators focused on product/service sentiment.