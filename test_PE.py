import os, sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'perception_models')))

# sys.path.append('../../../')
import decord

if torch.cuda.is_available():
    print('GPU is available. Use GPU for this script')
else:
    print('Use CPU for this demo')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


def get_PE_visual_feature(imaga_path, model):
    """
    Get visual feature of the image using PE model
    Args:
        model_name (str): name of the PE model
        imaga_path (str): path to the image
    Returns:
        torch.Tensor: visual feature of the image
    """
    # model_name = 'PE-Core-G14-448'


    preprocess = transforms.get_image_transform(model.image_size)
    # tokenizer = transforms.get_text_tokenizer(model.context_length)

    image = preprocess(Image.open(imaga_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
    return image_features

# model_name = "PE-Core-L14-336"
#
# model = pe.CLIP.from_config(model_name, pretrained=True)  # Downloads from HF
# model = model.to(device)
#
# preprocess = transforms.get_image_transform(model.image_size)
# tokenizer = transforms.get_text_tokenizer(model.context_length)
#
# image = preprocess(Image.open("./perception_models/apps/pe/docs/assets/cat.png")).unsqueeze(0).to(device)
# captions = ["a diagram", "a dog", "a cat"]
# text = tokenizer(captions).to(device)
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]

# plt.imshow(Image.open("./assets/cat.png"))
# plt.axis('off')
# plt.show()
# print("Captions:", captions)
# print("Label probs:", ' '.join(['{:.2f}'.format(prob) for prob in text_probs]))  # prints: [[0.00, 0.00, 1.00]]
# print(f"This image is about {captions[text_probs.argmax()]}")

if __name__ == "__main__":
    model_name = "PE-Core-L14-336"
    model = pe.CLIP.from_config(model_name, pretrained=True)  # Downloads from HF
    model = model.to(device)
    visual_feature = get_PE_visual_feature("./perception_models/apps/pe/docs/assets/cat.png", model)

    print(visual_feature.shape)