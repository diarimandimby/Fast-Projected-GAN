# Module for preparing dataset and Training Faster Projected GAN
# Author : Diarimandimby Riantsoa Kanto

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from generator import FasterProjectedGANGenerator
from discriminator import ProjectedGANDiscriminator
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn.functional as F
from torchvision.transforms.v2 import PILToTensor

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def createData(paths, output_res):
  pbar = tqdm(paths, desc="Prétraitements des images")
  nb_img = 0
  list_images = []
  for file in pbar:
    image = Image.open(file).convert("RGB").resize((output_res, output_res))

    list_images.append(PILToTensor()(image).to(torch.float32) / 127.5 - 1)

    nb_img = len(list_images)

    pbar.set_postfix({"list_len": f"{nb_img}"})
  
  return torch.stack(list_images).to(device)

def createDataLoader(paths, output_res, batch_size, shuffle=True):
  return DataLoader(
    createData(paths, output_res),
    batch_size=batch_size,
    shuffle=shuffle
  )

def train(
  n_epochs,
  batch_size,
  data_loader,
  checkpoint_path,
  log_path,
  output_res, 
  glr=0.0002, 
  dlr=0.0002, 
  betas=(0.0, 0.99), 
  eps=1e-8,
  p=1, 
  s=100,
  checkpoint_file=''
):
  discriminator = ProjectedGANDiscriminator().to(device)
  generator = FasterProjectedGANGenerator(output_res).to(device)
  opt_G = Adam(generator.parameters(), lr=glr, betas=betas, eps=eps)
  opt_D = Adam(discriminator.parameters(), lr=dlr, betas=betas, eps=eps)
  gen_ema = AveragedModel(generator, multi_avg_fn=get_ema_multi_avg_fn(decay=0.999))

  if(checkpoint_file != ''):
    checkpoint = torch.load(checkpoint_file)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])    
    gen_ema.load_state_dict(checkpoint['gen_ema'])
    opt_G.load_state_dict(checkpoint['opt_G'])
    opt_D.load_state_dict(checkpoint['opt_D'])
    k = checkpoint['epoch']
  else:
    k = 0
  
  test_latents = torch.randn(24, 256, 1, 1).to(device)
  
  for epoch in range(k, n_epochs):
    
    for i, real_images in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):

      # --------------------------------------------
      # Train discriminator
      # --------------------------------------------

      opt_D.zero_grad()

      real_logits = discriminator(real_images)

      latent_vectors = torch.randn(batch_size, 256, 1, 1).to(device)
      with torch.no_grad():
        fake_images = generator(latent_vectors)

      fake_logits = discriminator(fake_images.detach())

      Dreal_loss = torch.mean(F.relu(torch.ones_like(real_logits) - real_logits))
      Dreal_loss.backward()
      
      Dgen_loss = torch.mean(F.relu(torch.ones_like(fake_logits) + fake_logits))
      Dgen_loss.backward()

      opt_D.step()

      # --------------------------------------------
      # Train generator
      # --------------------------------------------

      opt_G.zero_grad()

      latent_vectors = torch.randn(batch_size, 256, 1, 1).to(device)
      fake_images = generator(latent_vectors)

      fake_logits = discriminator(fake_images)

      G_loss = -torch.mean(fake_logits)
      G_loss.backward()
      
      opt_G.step()
      
      gen_ema.update_parameters(generator)

    if((epoch + 1) % p == 0):

      with torch.no_grad():
        sample_images = gen_ema(test_latents)

      fig, axes = plt.subplots(4, 6, figsize=(12, 8))
      plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
      for nrow in range(4):
        for ncol in range(6):
          outputs = (sample_images[nrow + ncol] + 1) /2

          axes[nrow, ncol].imshow(outputs.cpu().permute(1, 2, 0).numpy())
          axes[nrow, ncol].axis('off')
      
      fig.savefig(f'{log_path}/{epoch + 1}.png', transparent=False, dpi=80, bbox_inches="tight")

    if((epoch + 1) % s == 0):
      torch.save({
        'epoch': epoch + 1,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'gen_ema': gen_ema.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
      }, f'{checkpoint_path}/Faster_pg_checkpoint_epoch_{epoch + 1}.pth')

    print(f"g_loss : {G_loss.item()}, dgen_loss : {Dgen_loss.item()}, dreal_loss : {Dreal_loss.item()}")
