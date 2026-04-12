import torch
import numpy as np
from dream import Branch, norm_angle

model_dream = Branch(feat_dim=512)
# model.cuda()
checkpoint = torch.load('checkpoint_512.pth')
model_dream.load_state_dict(checkpoint['state_dict'])
model_dream.eval()

def dream_embedding(embedding_I):
    yaw = np.zeros([1, 1])
    yaw[0,0] = norm_angle(float(0))
    original_embedding_tensor = np.expand_dims(embedding_I.detach().cpu().numpy(), axis=0)
    # feat = torch.autograd.Variable(torch.from_numpy(feat.astype(np.float32)), volatile=True).cuda()
    # yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)), volatile=True).cuda()
    feature_original = torch.autograd.Variable(torch.from_numpy(original_embedding_tensor.astype(np.float32)))
    yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)))

    new_embedding = model_dream(feature_original, yaw)
    # new_embedding = new_embedding.cpu().data.numpy()
    new_embedding = new_embedding.to(device).data.numpy()
    embedding_I = new_embedding[0, :]
    return embedding_I