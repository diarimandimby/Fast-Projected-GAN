import torch
from torch import nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from generator import FastGANGenerator
from discriminator import ProjectedGANDiscriminator
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.nn.functional as F
import random
from torchvision.transforms.v2 import GaussianBlur

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def createData(paths, output_res):
  transform = transforms.Compose([
    transforms.Resize((output_res, output_res)),
    transforms.ToTensor()
  ])
  
  pbar = tqdm(paths, desc="Prétraitements des images")
  nb_img = 0
  list_images = []
  for file in pbar:
    image = Image.open(file).convert("RGB")
    
    tensor_image = transform(image)
    
    list_images.append(2 * tensor_image - 1)

    nb_img = len(list_images)

    pbar.set_postfix({"list_len": f"{nb_img}"})
  
  return torch.stack(list_images).to(device)

def createDataLoader(paths, output_res, batch_size, shuffle=True):
  return DataLoader(
    createData(paths, output_res),
    batch_size=batch_size,
    shuffle=shuffle
  )

def addRandomGaussianNoise(tensor_images, noise_sigma_range):
  """ Ajoute du bruit. """
  color = random.randint(0, 1)
  sigma = random.uniform(noise_sigma_range[0], noise_sigma_range[1])
  if(color):
    t = tensor_images + torch.randn_like(tensor_images)*sigma
  else:
    noise = torch.randn(1, tensor_images.shape[2], tensor_images.shape[3])*sigma
    noise = noise.repeat(tensor_images.shape[0], 3, 1, 1)
    t = tensor_images + noise.to(device)
  return t

def addRandomGaussianBlur(tensor_images, sigma_range):
  kernel_size = random.randint(5, 21)
  if(kernel_size % 2 == 0):
    kernel_size += 1
  blur_transform = GaussianBlur(kernel_size=kernel_size, sigma=(sigma_range[0], sigma_range[1]))
  return blur_transform(tensor_images)

def train(
  n_epochs,
  batch_size,
  data_loader,
  checkpoint_path,
  output_res, 
  glr=0.0002, 
  dlr=0.0002, 
  betas=(0.0, 0.99), 
  eps=1e-8,
  pfr=1, 
  sfr=100, 
  checkpoint_file=''
):
  discriminator = ProjectedGANDiscriminator().to(device)
  generator = FastGANGenerator(output_res).to(device)
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
  
  test_latents = torch.randn(4, 256, 1, 1).to(device)
  
  for epoch in range(k, n_epochs):
    
    for i, real_images in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):

      # --------------------------------------------
      # Entraînement du Discriminateur
      # --------------------------------------------

      opt_D.zero_grad()

      real_images = addRandomGaussianNoise(real_images, [30, 50])
      real_images = addRandomGaussianBlur(real_images, [0.2, 3])

      real_logits = discriminator(real_images)

      latent_vectors = torch.randn(batch_size, 256, 1, 1).to(device)
      with torch.no_grad():
        fake_images = generator(latent_vectors)

      fake_images = addRandomGaussianNoise(fake_images, [30, 50])
      fake_images = addRandomGaussianBlur(fake_images, [0.2, 3])

      fake_logits = discriminator(fake_images.detach())

      Dreal_loss = torch.mean(F.relu(torch.ones_like(real_logits) - real_logits))
      Dgen_loss = torch.mean(F.relu(torch.ones_like(fake_logits) + fake_logits))

      D_loss = Dgen_loss + Dreal_loss

      D_loss.backward()

      opt_D.step()

      # --------------------------------------------
      # Entraînement du Générateur
      # --------------------------------------------

      opt_G.zero_grad()

      latent_vectors = torch.randn(batch_size, 256, 1, 1).to(device)
      fake_images = generator(latent_vectors)

      fake_images = addRandomGaussianNoise(fake_images, [30, 50])
      fake_images = addRandomGaussianBlur(fake_images, [0.2, 3])

      fake_logits = discriminator(fake_images)

      G_loss = -torch.mean(fake_logits)

      G_loss.backward()

      print(generator.initial_block[0].weight.grad)
      
      opt_G.step()
      gen_ema.update_parameters(generator)

    if((epoch + 1) % pfr == 0):

      with torch.no_grad():
        sample_images = gen_ema(test_latents)

      fig, axes = plt.subplots(1, 4, figsize=(8, 2))
      for ncol in range(4):
        outputs = (sample_images[ncol] + 1) /2

        axes[ncol].imshow(outputs.cpu().permute(1, 2, 0).numpy())
        axes[ncol].axis('off')
      plt.show()

    if((epoch + 1) % sfr == 0):
      torch.save({
        'epoch': epoch + 1,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'gen_ema': gen_ema.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
      }, f'{checkpoint_path}/Faster_pg_checkpoint_epoch_{epoch + 1}.pth')

    print(f"g_loss : {G_loss.item()}, dgen_loss : {Dgen_loss.item()}, dreal_loss : {Dreal_loss.item()}")
