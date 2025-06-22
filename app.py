import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ================== MODEL DEFINITION ==================
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.label_embed = nn.Embedding(10, 10)
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        y_emb = self.label_embed(y)
        x = torch.cat([x, y_emb], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.label_embed(y)
        z = torch.cat([z, y_emb], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    model = CVAE()
    model.load_state_dict(torch.load("cvae_mnist.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================== STREAMLIT UI ==================
st.title("🧠 Conditional VAE - MNIST Digit Generator")
st.write("Generate multiple handwritten-style digits conditioned on the number you choose.")

digit = st.number_input("Choose a digit (0–9):", min_value=0, max_value=9, step=1)
samples = st.slider("Number of images to generate", min_value=1, max_value=10, value=5)

if st.button("Generate"):
    with torch.no_grad():
        y = torch.tensor([digit] * samples, dtype=torch.long)  # Ensure correct type
        z = torch.randn(samples, 20)
        generated = model.decode(z, y).view(-1, 28, 28).numpy()

        fig, axes = plt.subplots(1, samples, figsize=(samples * 1.5, 2))
        for i in range(samples):
            axes[i].imshow(generated[i], cmap="gray")
            axes[i].axis("off")
        st.pyplot(fig)
