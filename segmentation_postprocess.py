import torch
import matplotlib.pyplot as plt

imgs = torch.load('imgs.pt')
seg = torch.load('seg.pt')
pseg = torch.load('pseg.pt')

print(imgs[0][0].shape)
print(seg.shape)
print(pseg.shape)

#gt1 = torch.argmax(pseg[0], dim=0).detach().cpu().numpy()
#plt.imshow(gt1, cmap='gray')
#plt.savefig("gt1.png")