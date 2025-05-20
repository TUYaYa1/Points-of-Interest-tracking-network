import gc
import os
import numpy as np
import torch
from torch import nn
from models.unet_cbam import U_Net_v1

weight_path = 'your path'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
test_num = 'your num'
batch_size = 2
net = U_Net_v1().to(device)
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight！')
else:
    print('not successful load weight')
loss_f = nn.BCELoss().to(device)

def test():
    net.to(device)
    net.eval()
    zuobiao = np.zeros([0,3,2])
    zuobiao_w1 = np.zeros([3, 1])
    zuobiao_h1 = np.zeros([3, 1])
    zuobiao2 = np.zeros([2,3,2])

    img = np.load('your path')
    label = np.load('your path')

    print(img.shape, label.shape)

    state = np.random.get_state()
    np.random.shuffle(img)
    np.random.set_state(state)
    np.random.shuffle(label)

    # 1000,1,128,128
    img_tensor = torch.FloatTensor(img).unsqueeze(1).to(device)
    # 1000,3,128,128
    label_tensor = torch.FloatTensor(label).to(device)
    print(img_tensor.shape)
    print(label_tensor.shape)

    test_img = img_tensor[900:1000]
    test_label = label_tensor[900:1000]

    sum_loss = 0
    s = 0
    for b in range(0, test_num, batch_size):
        start_idx = b
        end_idx = (b + batch_size) if (b + batch_size) < test_num else test_num
        #
        test_input_imgA = test_img[start_idx:end_idx]

        with torch.no_grad():
            test_out = net(test_input_imgA).to(device)
            print(test_out.shape)
            out=test_out.cpu().detach().numpy()
            for i in range(2):
                for j in range(3):
                    h, w = np.where(out[i][j] == out[i][j].max())
                    zuobiao_w1[j,...] = w * 1536 / 768
                    zuobiao_h1[j,...] = h * 1536 / 768
                    zuobiao1 = np.concatenate((zuobiao_w1, zuobiao_h1), axis=1)
                    print(zuobiao1.shape)
                # zuobiao_w2[i,...] = zuobiao_w1
                # zuobiao_h2[i,...] = zuobiao_h1
                zuobiao2[i,...] = zuobiao1
            # zuobiao_w = np.concatenate((zuobiao_w, zuobiao_w2), axis=0)
            # zuobiao_h = np.concatenate((zuobiao_h, zuobiao_h2), axis=0)
            zuobiao = np.concatenate((zuobiao, zuobiao2), axis=0)
            print(zuobiao.shape)
            torch.cuda.empty_cache()
    np.save("save path",zuobiao)
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    test()