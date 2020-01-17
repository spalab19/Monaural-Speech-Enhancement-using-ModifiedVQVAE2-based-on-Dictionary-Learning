import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # dim_embedding --> D(=C), num_embeddings --> K
        self.num_embedding = K
        self.dim_embedding = D
        self.embedding = nn.Embedding(K, D)
        # weight initialization
        self.embedding.weight.data.normal_()
        # for EMA parameter
        self.register_buffer('cluster_size', torch.zeros(K))
        self.ema_w = nn.Parameter(torch.Tensor(K, D))
        self.ema_w.data.normal_()
        self.decay = 0.99
        self.eps = 1e-5

    def forward(self, z_e_x):
        # Convert inputs from BCHW -> BHWC
        inputs = z_e_x.permute(0, 2, 3, 1).contiguous()
        # flatten inputs
        flatten = inputs.reshape(-1, self.dim_embedding)
        # Compute the distances to the codebook
        distances = (torch.sum(flatten**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flatten, self.embedding.weight.t()))
        # Encoding
        self.encoding_idx = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(self.encoding_idx.shape[0],
                                self.num_embedding).to(device)
        # Make one-hot vector
        encodings.scatter_(1, self.encoding_idx, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self.cluster_size = self.cluster_size * self.decay + \
                                      (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self.cluster_size.data)
            self.cluster_size = (
                (self.cluster_size+self.eps) / (n+self.num_embedding*self.eps)*n)
            embedding_sum = torch.matmul(encodings.t(), flatten)
            self.ema_w = nn.Parameter(self.ema_w * self.decay
                                       + (1 - self.decay) * embedding_sum)
            self.embedding.weight = nn.Parameter(self.ema_w
                                                 / self.cluster_size.unsqueeze(1))
        # Quantize and unflatten
        quantize = torch.matmul(encodings, self.embedding.weight).view_as(inputs)
        # Latent loss
        latent_loss = torch.mean((quantize.detach() - inputs)**2)

        quantize = inputs + (quantize - inputs).detach()
        # revise BHWC --> BCHW, and input of decoder
        return quantize.permute(0, 3, 1, 2).contiguous(), latent_loss


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim//4, 3, 1, 1),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(True),
            nn.Conv2d(dim//4, dim, 1, 1, 0),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

class Transpose(nn.Module):
    def forward(self, x, dim=[0, 2, 1, 3]):
        return x.permute(*dim)

class Filterencoder(nn.Module):
    def __init__(self, f_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, dim, (f_dim, 1), (1, 1), (0, 0)),
            nn.BatchNorm2d(dim),
            nn.ReLU(True)
        )
    def forward(self, x):
        return self.block(x)

class Filterdecoder(nn.Module):
    def __init__(self, f_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(dim, 1, (f_dim, 1), (1, 1), (0, 0))
        )
    def forward(self, x):
        return self.block(x)


# (Input + 2* pad - kernel) / stride +1 = output
class Encoder(nn.Module):
    def __init__(self, in_dim, dim, stride):
        super().__init__()
        if stride == 4:
            self.block = nn.Sequential(
                Transpose(),
                nn.Conv2d(in_dim, dim//2, 4, 2, 1),
                nn.BatchNorm2d(dim//2),
                nn.ReLU(True),
                nn.Conv2d(dim//2, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 3, 1, 1),
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True)
            )
        elif stride == 2:
            self.block = nn.Sequential(
                nn.Conv2d(in_dim, dim//2, 4, 2, 1),
                nn.BatchNorm2d(dim//2),
                nn.ReLU(True),
                nn.Conv2d(dim//2, dim, 3, 1, 1),
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True)
            )
    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, dim, stride):
        super().__init__()
        if stride == 4:
            self.block = nn.Sequential(
                nn.Conv2d(in_dim, dim, 3, 1, 1),
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim//2, 4, 2, 1),
                nn.BatchNorm2d(dim//2),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim//2, out_dim, 4, 2, 1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
                Transpose()
            )
        elif stride == 2:
            self.block = nn.Sequential(
                nn.Conv2d(in_dim, dim, 3, 1, 1),
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, out_dim, 4, 2, 1)
            )
    def forward(self, x):
        return self.block(x)


class VQVAE2(nn.Module):
    def __init__(self, f_dim, dim, K, D):
        super(VQVAE2, self).__init__()
        self.filter_e = Filterencoder(f_dim, dim//2)

        self.encoder_b = Encoder(1, dim, stride=4)
        self.encoder_t = Encoder(dim, dim, stride=2)

        self.pre_quantize_conv_t = nn.Conv2d(dim, D, 1, 1, 0)
        self.quantize_t = VQEmbedding(K, D)

        self.decoder_t = Decoder(D, D, dim, stride=2)

        self.pre_quantize_conv_b = nn.Conv2d(D+dim, D, 1, 1, 0)
        self.quantize_b = VQEmbedding(K, D)

        self.upsample_t = nn.ConvTranspose2d(D, D, 4, 2, 1)
        self.decoder = Decoder(D+D, 1, dim, stride=4)

        self.decoder_b = Decoder(D, 1, dim, stride=4)

        self.filter_d = Filterdecoder(f_dim, dim)

    def forward(self, x):
        x_f = self.filter_e(x)
        z_e_b = self.encoder_b(x_f)
        z_e_t = self.encoder_t(z_e_b)

        z_q_t = self.pre_quantize_conv_t(z_e_t)
        z_q_t, latent_loss_t = self.quantize_t(z_q_t)
        latent_loss_t = latent_loss_t.unsqueeze(0)

        z_d_t = self.decoder_t(z_q_t)
        z_e_b = torch.cat([z_d_t, z_e_b], 1)

        z_q_b = self.pre_quantize_conv_b(z_e_b)
        z_q_b, latent_loss_b = self.quantize_b(z_q_b)
        latent_loss_b = latent_loss_b.unsqueeze(0)

        z_q_x = self.upsample_t(z_q_t)
        z_q_x = torch.cat([z_q_x, z_q_b], 1)
        x_b = self.decoder(z_q_x)

        x_c = torch.cat([x_f, x_b], 1)

        x_tilde = self.filter_d(x_c)

        return x_tilde, latent_loss_t+latent_loss_b
