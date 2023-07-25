from modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
import numpy as np
from sklearn.decomposition import PCA
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')
print("test2", flush=True)


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = join(root)
        self.transform = transform
        self.images = os.listdir("/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/train/rgb/virtual")

    def __getitem__(self, index):
        image = Image.open(join("/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/train/rgb/virtual", self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


@hydra.main(config_path="configs", config_name="demo_config.yml")
def my_app(cfg: DictConfig) -> None:
    result_dir = "/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/train"
    os.makedirs(join(result_dir, "rel_new/virtual"), exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    print(OmegaConf.to_yaml(model.cfg))

    dataset = UnlabeledImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.res, False, None),
    )

    loader = DataLoader(dataset, cfg.batch_size * 2,
                        shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    # pca = PCA(n_components=32)
    with open('pca_virtual.obj', 'rb') as input_file:
        pca = pickle.load(input_file)
    # data = np.zeros((250000000, 90), dtype='uint8')
    # idx = 0
    cnt = 0
    for i, (img, name) in enumerate(loader):
        with torch.no_grad():
            img = img.cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            arr = code.cpu().numpy().transpose(0, 2, 3, 1)
            s = arr.shape
            arr = pca.transform(arr.reshape(s[0] * s[1] * s[2], s[3]))
            arr = arr.reshape(s[0], s[1], s[2], 32)

            mins = np.zeros((32), dtype='float32')
            maxs = np.zeros((32), dtype='float32')

            for j in range(img.shape[0]):
                cnt += 1
                new_name = ".".join(name[j].split(".")[:-1])
                res = arr[j]
                res = res.reshape(s[1] * s[2], 32)
                mins = np.minimum(mins, res.min(axis=0))
                maxs = np.maximum(maxs, res.max(axis=0))
            if cnt > 1000:
                break
    print(mins, flush=True)
    print(maxs, flush=True)

if __name__ == "__main__":
    prep_args()
    my_app()
