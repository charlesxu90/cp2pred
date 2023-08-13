import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt



def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def _exclude_layer(layer):
    if isinstance(layer, nn.Sequential):
        return True
    if not 'torch.nn' in str(layer.__class__):
        return True

    return False

def choose_tlayer(model):
    name_to_num = {}
    num_to_layer = {}
    for idx, data in enumerate(model.named_modules()):
        name, layer = data
        if _exclude_layer(layer):
            continue

        name_to_num[name] = idx
        num_to_layer[idx] = layer
        print(f'[ Number: {idx},  Name: {name} ] -> Layer: {layer}\n')

    a = input(f'Choose "Number" or "Name" of a target layer: ')

    if a.isnumeric() == False:
        a = name_to_num[a]
    else:
        a = int(a)
    try:
        t_layer = num_to_layer[a]
        return t_layer
    except:
        raise Exception(f'Selected index {a} is not allowed.')


class GradCAM():
    def __init__(self, model, thresh=0.0, select_t_layer=False):
        self.model = model
        self.select_t_layer = select_t_layer
        self.thresh = thresh

        # Save outputs of forward and backward hooking
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        # find finalconv layer name
        if self.select_t_layer == False:
            finalconv_after = ['classifier', 'avgpool', 'fc']

            for idx, m in enumerate(self.model._modules.items()):
                if any(x in m for x in finalconv_after):
                    break
                else:
                    self.finalconv_module = m[1]

            # get a last layer of the module
            self.t_layer = self.finalconv_module[-1]
        else:
            # get a target layer from user's input
            self.t_layer = choose_tlayer(self.model)

        self.t_layer.register_forward_hook(forward_hook)
        # self.t_layer.register_backward_hook(backward_hook)
        self.t_layer.register_full_backward_hook(backward_hook)

    def __call__(self, img, img_size=224, class_index=None):
        assert img[1].shape[0] == 1
        img_show, img = img[0], img[1]
        input = img.reshape(-1, 3, img_size, img_size).cuda()
        output = self.model(input)

        if class_index == None:  # get class index of highest probability
            class_index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][class_index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)

        if cuda_available():
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.gradients['value']
        activations = self.activations['value']

        # reshaping
        weights = torch.mean(torch.mean(gradients, dim=2), dim=2)
        weights = weights.reshape(weights.shape[1], 1, 1)
        activationMap = torch.squeeze(activations[0])

        # Get gradcam
        gradcam = F.relu((weights * activationMap).sum(0))
        gradcam = cv2.resize(gradcam.data.cpu().numpy(), (img_size, img_size))

        return gradcam


def get_image(img_path, normalize, img_transformer):

    img = Image.open(img_path).convert('RGB')
    img_trans = img_transformer(img)
    img_show = img_trans.numpy().transpose(1, 2, 0)

    return img_show, normalize(img_trans)

def get_transformed_img(img_path, img_size=224):
    img_cuda = torch.FloatTensor(1, 3, img_size, img_size).cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transformer = [transforms.CenterCrop(img_size), transforms.ToTensor()]
    img_show, img = get_image(img_path, normalize, transforms.Compose(img_transformer))
    img_cuda.copy_(torch.unsqueeze(img, 0))
    return img_cuda, img_show

def get_gradcam_plus_img(mask, img, img_size=224, thresh=.0):
    mask = (mask - np.min(mask)) / np.max(mask)
    img = img.reshape(img_size, img_size, 3)
    new_mask = mask.copy()
    new_mask[new_mask < thresh] = 0
    heatmap = cv2.applyColorMap(np.uint8(255 * new_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    gradcam = 1.0 * heatmap + img
    gradcam = gradcam / np.max(gradcam)
    return gradcam, heatmap

def get_gradcam(gradcam_obj, img_path, img_size=224):
    img_cuda, img_show = get_transformed_img(img_path, img_size=img_size)
    gradcam = gradcam_obj(img=(img_show, img_cuda), img_size=img_size)
    img_gradcam, _ = get_gradcam_plus_img(gradcam, img_show, img_size=img_size, thresh=.0)
    return img_gradcam, img_show

def save_gradcam(img_gradcam, gradcam_path,):
    try:
        assert cv2.imwrite(gradcam_path, np.uint8(255 * img_gradcam))
    except:
        Exception(f"Failed to save gradcam to {gradcam_path}")

def show_gradcam(img_show, img_gradcam):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('GradCam')
    _ = ax1.imshow(img_show)
    _ = ax2.imshow(img_gradcam)

def run_gradcam(img_path, model, img_size=224):
    gradcam_obj = GradCAM(model=model, thresh=0.0)
    img_gradcam, img_show = get_gradcam(gradcam_obj, img_path, img_size=img_size)
    save_gradcam(img_gradcam, img_path + '_gradcam.png',)
    show_gradcam(img_show, img_gradcam)


if __name__ == '__main__':
    from model.resnet import init_model, load_model_from_ckpt
    from utils.utils import count_params

    model = init_model("ResNet18", num_classes=1)
    model = load_model_from_ckpt("ResNet18", model, 'data/resnet_ckpt/ImageMol.pth.tar')

    model = model.to('cuda')
    model.eval()
    count_params(model)  # 11 million parameters
    run_gradcam("data/gradcam/cycpep_0.png", model, img_size=224)
    # run_gradcam("data/gradcam/1.png", model)
