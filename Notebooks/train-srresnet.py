import tensorflow as tf
from dataset import download_data
from models import SRResNet
from train import train_edsr_srresnet
from tqdm import tqdm

train_ds, val_ds = download_data()

model = SRResNet()
trainer = train_edsr_srresnet(model)

for epoch in range(10):

    for lr, hr in train_ds.take(1000):
        loss_value, psnr_value = trainer.train_step(lr, hr)

    if epoch % 10 == 0 or epoch == 199:
        print(f'Epochs : {epoch}   ||   Loss : {loss_value:.5f}   ||   PSNR : {psnr_value:.5f}')

# model.save('saved_model/',save_format='tf')
model.save("model.h5")