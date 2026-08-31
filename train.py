from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

def transform():
  return transforms.Compose([
    transforms.Resize((output_res, output_res)),
    transforms.ToTensor()
  ])

def createData(paths):
  pbar = tqdm(paths, desc="Prétraitements des images")
  nb_img = 0
  list_images = []
  for file in pbar:
    image = Image.open(file).convert("RGB")

    tensor_image = transform(image).to(device)

    list_images.append(2 * tensor_image - 1)

    nb_img = len(list_images)

    pbar.set_postfix({"list_len": f"{nb_img}"})
  
  tensor_images = torch.stack(list_images)

def dataLoader(dataset, batch_size, shuffle=True):
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle
  )

def train(n_epochs, data_loader):
  test_latents = torch.randn(4, 256, 1, 1).to(device)
  for epoch in range(n_epochs):
    for i, real_images in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):

      # Définir une variance qui décroît au fil des époques (ex: démarre à 0.2, baisse vers 0)
      decaying_noise = 0.5 * (0.9999 ** epoch)

      # --------------------------------------------
      # Entraînement du Discriminateur
      # --------------------------------------------

      opt_D.zero_grad()

      if decaying_noise > 0:
        # On ajoute le bruit uniquement pour le calcul des logits du Discriminateur
        real_images = real_images + torch.randn_like(real_images) * decaying_noise

      real_logits = discriminator(randomGaussianBlur(real_images))

      # B. Générer de fausses images et les passer dans l'extracteur figé
      latent_vectors = torch.randn(batch_size, 256, 1, 1).to(device)  # Vecteur latent du GAN

      with torch.no_grad():
        fake_images = generator(latent_vectors)

      if decaying_noise > 0:
        # On ajoute le bruit uniquement pour le calcul des logits du Discriminateur
        fake_images = fake_images + torch.randn_like(fake_images) * decaying_noise

      fake_logits = discriminator(randomGaussianBlur(fake_images.detach()))

      del fake_images

      # C. Calcul de la perte et rétropropagation
      Dreal_loss = torch.mean(F.relu(torch.ones_like(real_logits) - real_logits))
      Dgen_loss = torch.mean(F.relu(torch.ones_like(fake_logits) + fake_logits))

      D_loss = Dgen_loss + Dreal_loss

      D_loss.backward()

      opt_D.step()


      # --------------------------------------------
      # Entraînement du Générateur
      # --------------------------------------------

      opt_G.zero_grad()

      latent_vectors = torch.randn(batch_size, 256, 1, 1).to(device)  # Vecteur latent du GAN
      fake_images = generator(latent_vectors)

      # On réanalyse les fausses images, mais cette fois sans détacher le Générateur
      fake_logits = discriminator(fake_images)

      # Le générateur tente de tromper tous les discriminateurs à la fois
      G_loss = -torch.mean(fake_logits)

      G_loss.backward()
      opt_G.step()

      gen_ema.update_parameters(generator)


    if epoch % 2 == 0:

      with torch.no_grad():
        sample_images = gen_ema(test_latents)

      # --- BLOC DE DIAGNOSTIC ---
      if 'old_samples' in locals():
        diff = torch.abs(sample_images - old_samples).mean().item()
        print(f"\n[DIAGNOSTIC] Différence numérique réelle avec l'époque précédente : {diff}")
      old_samples = sample_images.clone()

      fig, axes = plt.subplots(1, 4, figsize=(8, 2))
      for ncol in range(4):
        outputs = (sample_images[ncol] + 1) /2

        axes[ncol].imshow(outputs.cpu().permute(1, 2, 0).numpy())
        axes[ncol].axis('off')
      plt.show()

  print(f"g_loss : {G_loss.item()}, dgen_loss : {Dgen_loss.item()}, dreal_loss : {Dreal_loss.item()}")
