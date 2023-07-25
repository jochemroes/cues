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
        self.images = os.listdir("/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/train/rgb/eigen")

    def __getitem__(self, index):
        image = Image.open(join("/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/train/rgb/eigen", self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


@hydra.main(config_path="configs", config_name="demo_config.yml")
def my_app(cfg: DictConfig) -> None:
    # result_dir = "/deepstore/datasets/dmb/ComputerVision/nis-data/jochem/train"
    # os.makedirs(join(result_dir, "rel_vec/eigen"), exist_ok=True)

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

    pca = PCA(n_components=32)
    data = np.zeros((2500000, 90), dtype='float32')
    idx = 0
    for i, (img, name) in enumerate(loader):
        with torch.no_grad():
            img = img.cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            add = code[0]
            st = np.random.randint(0, 102400)
            en = np.random.randint(st, min(st + 10000, 102400))
            if idx + en - st > data.shape[0]:
                break
            arr = add.cpu().numpy().transpose(1, 2, 0)
            s = arr.shape
            data[idx:idx+en-st] = arr.reshape(s[0] * s[1], s[2])[st:en]
            idx += en - st

    data = data[:idx, :]
    pca.fit(data)
    with open('pca_eigen.obj', 'wb') as f:
        pickle.dump(pca, f)


if __name__ == "__main__":
    prep_args()
    my_app()
