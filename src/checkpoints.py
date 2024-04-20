from pytorch_lightning.callbacks import ModelCheckpoint

val_checkpoint = ModelCheckpoint(
    monitor='val_jaccard',
    dirpath='checkpoints/',
    filename='e{epoch:02d}-iou{val_jaccard:.2f}',
    auto_insert_metric_name=False,
    save_last=True,
    mode='max',
)

regular_checkpoint = ModelCheckpoint(
    monitor='epoch',
    dirpath='checkpoints/',
    filename='latest-e{epoch:02d}',
    auto_insert_metric_name=False,
    mode='max',
    every_n_epochs=1,
    save_last=True,
)
