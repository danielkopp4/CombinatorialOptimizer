import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import random

manualSeed = 1403
workers = 2
batch_size = 128

class Model(torch.nn.Module):
    def __init__(self, input_features, n):
        super(Model, self).__init__()
        self.lin1 = torch.nn.Linear(input_features, input_features * 10)
        self.lin2 = torch.nn.Linear(input_features * 10, n * 10)
        self.lin3 = torch.nn.Linear(n * 10, n * n * 2)
        self.lin4 = torch.nn.Linear(input_features * 10, n * n * 9)
        self.conv1 = torch.nn.Conv1d(3 * n, 2 * n, 3, 1, 1)
        self.conv2 = torch.nn.Conv1d(2 * n, 3 * n // 2, 3, 1, 0)
        self.conv3 = torch.nn.Conv1d(3 * n // 2, n, 3, 1, 0)
        self.conv4 = torch.nn.Conv1d(n, n, 3, 1, 0)
        self.conv5 = torch.nn.Conv1d(n, n, 3, 1, 0)
        self.conv6 = torch.nn.Conv1d(n, n, 3, 1, 0)
        self.h_sig = torch.nn.Hardsigmoid()
        self.l_relu = torch.nn.LeakyReLU(0.2)
        self.n = n

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        # x = self.lin2(x)
        # x = self.l_relu(x)
        # x = self.lin3(x)
        # x = self.l_relu(x)
        x = self.lin4(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), 3 * self.n, 3 * self.n)
        x = self.conv1(x)
        x = self.l_relu(x)
        x = self.conv2(x)
        x = self.l_relu(x)
        # x = F.relu(x)
        x = self.conv3(x)
        x = self.l_relu(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # print(x.size())
        # x = self.softmax(x)
        # x = torch.ceil(x - 0.5)
        # x = torch.sigmoid(x)
        # x = torch.floor(x + 0.5)
        x = self.h_sig(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, n, ndf=64):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(n, ndf, 4, 2, 2, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(ndf, ndf * 2, 4, 2, 2, bias=False),
            torch.nn.BatchNorm1d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 2, bias=False),
            torch.nn.BatchNorm1d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 4, bias=False),
            torch.nn.BatchNorm1d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv1d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# def make_y(x_in):
#     x = x_in.clone()
#     x[x < 0.5] =  0
#     x[x >= 0.5] = 1

#     for i in range(x.shape[0]):
#         row_s = torch.sum(x[i])
#         if row_s == 0:
#             for j in range(x.shape[1]):
#                 col_s = torch.sum(x[:, j])

#                 if col_s == 0:
#                     x[i][j] = 1
#                     break
#     return x

def genrate_rand_permutation_mat(n):
    order = np.arange(n)
    np.random.shuffle(order)

    out = np.zeros((n, n), np.float32)

    for i in range(len(order)):
        out[i][order[i]] = 1

    return torch.from_numpy(out).view(n, n)

def generate_m_rand_perm_mat(m, n):
    ret = torch.zeros((m, n, n))
    for i in range(m):
        ret[i] = genrate_rand_permutation_mat(n)

    ret.requires_grad = True
    return ret

def norm_to_nn(x_in):
    x = x_in.clone()
    x *= 2
    x -= 1
    return x

def nn_out_to_norm(x_in):
    x = x_in.clone()
    x += 1
    x *= 2
    return x

# img_list = []
# G_losses = []
# D_losses = []
# iters = 0

# print("Starting Training Loop...")
# # For each epoch
# for epoch in range(num_epochs):
#     # For each batch in the dataloader
#     for i, data in enumerate(dataloader, 0):
#         ############################
#         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#         ###########################
#         ## Train with all-real batch
#         netD.zero_grad()
#         # Format batch
#         real_cpu = data[0].to(device)
#         b_size = real_cpu.size(0)
#         label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
#         # Forward pass real batch through D
#         output = netD(real_cpu).view(-1)
#         # Calculate loss on all-real batch
#         errD_real = criterion(output, label)
#         # Calculate gradients for D in backward pass
#         errD_real.backward()
#         D_x = output.mean().item()

#         ## Train with all-fake batch
#         # Generate batch of latent vectors
#         noise = torch.randn(b_size, nz, 1, 1, device=device)
#         # Generate fake image batch with G
#         fake = netG(noise)
#         label.fill_(fake_label)
#         # Classify all fake batch with D
#         output = netD(fake.detach()).view(-1)
#         # Calculate D's loss on the all-fake batch
#         errD_fake = criterion(output, label)
#         # Calculate the gradients for this batch
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#         # Add the gradients from the all-real and all-fake batches
#         errD = errD_real + errD_fake
#         # Update D
#         optimizerD.step()

#         ############################
#         # (2) Update G network: maximize log(D(G(z)))
#         ###########################
#         netG.zero_grad()
#         label.fill_(real_label)  # fake labels are real for generator cost
#         # Since we just updated D, perform another forward pass of all-fake batch through D
#         output = netD(fake).view(-1)
#         # Calculate G's loss based on this output
#         errG = criterion(output, label)
#         # Calculate gradients for G
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         # Update G
#         optimizerG.step()

#         # Output training stats
#         if i % 50 == 0:
#             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                   % (epoch, num_epochs, i, len(dataloader),
#                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

#         # Save Losses for plotting later
#         G_losses.append(errG.item())
#         D_losses.append(errD.item())

#         # Check how the generator is doing by saving G's output on fixed_noise
#         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#             with torch.no_grad():
#                 fake = netG(fixed_noise).detach().cpu()
#             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

#         iters += 1


def train(n, epochs=100, samples_per_epoch=1000):
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    rand_noise_features = 10
    generator = Model(rand_noise_features, n)
    discriminator = Discriminator(n)
    print(generator)

    real_label = 1.
    fake_label = 0.


    lr = 5E-6
    beta1 = 0.5

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0


    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.01, betas=(beta1, 0.999))

    criterion = torch.nn.BCELoss()
       
    softmax = torch.nn.Softmax(1)

    for epoch in range(epochs):
        data_real = generate_m_rand_perm_mat(samples_per_epoch, n)
        data_inpt = softmax(torch.randn((samples_per_epoch, rand_noise_features)))
        data_inpt.requres_grad = True

        # for i in range(samples_per_epoch):

        # # target.requires_grad = True
        # optimizer.zero_grad()

        # output = model(inpt)
        # # print(target)
        # # input()

        # if i == itrs - 1:
        #     print(output)


        # loss = criterion(output, target)

        # # print(loss)
        # if i % 100 == 0:
        #     if i % 1000 == 0:
        #         print(output)
        #     # for x in model.parameters():
        #         # print(x)
        #     print(loss)

        # loss.backward()
        # optimizer.step()

        discriminator.zero_grad()
        # Format batch
        # real_cpu = data[0].to(device)
        # print(data_real)
        # print(data_inpt)
        b_size = data_real.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float)
        # Forward pass real batch through D
        output = discriminator(data_real).view(-1)
        # print(output[0])
        # print(label[0])
        # exit()
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(data_inpt)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # if 50 % 50 == 0:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch, epochs,
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if (epoch + 1) % 5 == 0:
            print(fake[0])

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_noise).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1



if __name__ == '__main__':
    train(5)