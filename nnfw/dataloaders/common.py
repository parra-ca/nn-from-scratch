from __future__ import annotations
from typing import Any, Sized, overload, override
import pathlib, os, torch, shutil
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, v2
import kornia.augmentation as K

from .. import plotter

_DATASETS_ROOT_PATH = "datasets"
_EVAL_BATCH_SZ = 1024 # batch size used when simply evaluating

class Loader():
    def __init__(
            self,
            ds_train_full: Dataset,
            ds_test: Dataset,
            data_params: dict[str, Any],
            name: str,
    ):
        # partition full train data into train + validation
        batch_sz = data_params["batch_sz"]
        val_split = data_params["val_split"]
        train_sz = len(ds_train_full)
        valid_sz = int(train_sz * val_split)
        train_sz = train_sz - valid_sz
        ds_train, ds_valid = random_split(ds_train_full, [train_sz, valid_sz])
        self.dl_train: DataLoader = DataLoader(ds_train, batch_size=batch_sz, shuffle=True)
        self.dl_valid: DataLoader = DataLoader(ds_valid, batch_size=_EVAL_BATCH_SZ,
                                   shuffle=False)

        # random sample from training data to monitor overfit
        tr_sample_sz = valid_sz
        idx = torch.randperm(train_sz)[:tr_sample_sz].tolist()
        tr_sample = Subset(ds_train, idx)
        self.dl_tr_sample: DataLoader = DataLoader(tr_sample, batch_size=_EVAL_BATCH_SZ,
                                       shuffle=False)
        
        # testing data
        self.dl_test: DataLoader = DataLoader(ds_test, batch_size=_EVAL_BATCH_SZ,
                                  shuffle=False)

        # compose augmentation function
        #self._augment_fn: v2.Transform | None = None
        self._augment_fn: K.AugmentationSequential | None = None
        if "augm_params" in data_params and \
           data_params["augm_params"] is not None:
            augm_params = data_params["augm_params"]
            augm_steps = []
            avail_transf = {
                "rotate" : lambda p: K.RandomRotation(**p),
                "elastic": lambda p: K.RandomElasticTransform(**p),
                "crop"   : lambda p: K.RandomCrop(**p),
                "hflip"  : lambda p: K.RandomHorizontalFlip(**p),
                "affine" : lambda p: K.RandomAffine(**p),
                "color"  : lambda p: K.ColorJitter(**p),
            }
            for key, params in augm_params.items():
                if key in avail_transf:
                    augm_steps.append(avail_transf[key](params))
                else:
                    print(f"Note: Given key ({key}) does not match any available "
                          "transformation.")
            self._augment_fn = K.AugmentationSequential(*augm_steps, same_on_batch=False)
            
        # save other metadata
        self.name: str = name
        self.labels: dict[int, str] = dict([(i,c) for i,c in enumerate(ds_train_full.classes)])
        self.shape = ds_train_full[0][0].shape
        return

    @classmethod
    def create(cls, data_params) -> Loader:
        raise NotImplementedError("Subclasses must implement create()")
        
    def augment(self, img_batch: Tensor) -> Tensor:
        if self._augment_fn is None:
            return img_batch
        return self._augment_fn(img_batch)
    
    def export_sample(self, sample_count: int = 10, sample_dir: str | None = None
                      ) -> str:
        # remove old exported images
        sample_dir = "sample_"+self.name if sample_dir is None else sample_dir
        os.makedirs(sample_dir, exist_ok=True)
        for p in pathlib.Path(sample_dir).iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

        count = 0
        augm = self._augment_fn is not None
        train_iterator = iter(self.dl_train)
        while count < sample_count:
            try:
                batch: tuple[Tensor,Tensor] = next(train_iterator)
            except StopIteration:
                break
            imgs_augm: Tensor | None = None
            if augm:
                imgs_augm = self.augment(batch[0])

            for i,(img,target) in enumerate(zip(*batch)):
                if count == sample_count:
                    break
                if target.numel() > 1:
                    target = torch.argmax(target)
                target = int(target.item())
                filename = f"{sample_dir}/{count:05}_{target}.png"
                title = self.labels[target]
                img_a = None
                if self.shape[0] == 1: # one channel
                    cmap_val = "gray"
                    img = img.squeeze()
                    if imgs_augm is not None:
                        img_a = imgs_augm[i].squeeze()
                elif self.shape[0] == 3: # three channels
                    cmap_val = None
                    img = img.permute(1,2,0)
                    if imgs_augm is not None:
                        img_a = imgs_augm[i].permute(1,2,0)
                else:
                    raise ValueError("Unexpected number of input channels "
                                     f"(Loader.shape[0]={self.shape[0]})")
                imgs = (img,)
                if img_a is not None:
                    imgs = (img,img_a)
                dpi = 5 * self.shape[-1] # dpi based on width
                print("dpi", dpi)
                plotter.export_sample(imgs, filename, title, cmap_val, dpi=dpi)
                count += 1
        return sample_dir

    def describe(self):
        """Shows the shape and dtype of the training, validation and test
        datasets."""
        names = ("Training", "Validation", "Testing")
        parts = (self.dl_train, self.dl_valid, self.dl_test)
        sized = isinstance(self.dl_train.dataset, Sized)
        tot_size = "???"
        if sized:
            tot_size = str(sum((
                len(self.dl_train.dataset),
                len(self.dl_valid.dataset),
                len(self.dl_test.dataset))))
        print(f"Dataset           : {self.name} ({tot_size} samples)")
        print(f"Online augmenting : {self._augment_fn is not None}")
        for name, dl in zip(names, parts):
            sz = "?"
            X, Y = dl.dataset[0]
            if isinstance(Y, Tensor):
                cl_shape = str(list(Y.shape))
                cl_type = str(Y.dtype)
            else:
                cl_shape = str(len(self.labels))
                cl_type = str(type(Y))
            cl_shape_width = len(cl_shape)
            
            w = max(len(str(list(X.shape))),cl_shape_width)
            if sized:
                sz = len(dl.dataset)
            print(f"{name}:\n"
                  f"    samples  : {sz}\n"
                  f"    batch_sz : {dl.batch_size}\n"
                  f"    features : {str(list(X.shape)):{w}} {str(X.dtype):<14}\n"
                  f"    classes  : {cl_shape:{w}} {cl_type:<14}")
        return

class MnistLoader(Loader):
    @classmethod
    @override
    def create(cls,data_params: dict[str, Any]) -> Loader:
        data_train = datasets.MNIST(
            root=_DATASETS_ROOT_PATH,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        data_test = datasets.MNIST(
            root=_DATASETS_ROOT_PATH,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return cls(data_train, data_test, data_params, "mnist")

class FashionMnistLoader(Loader):
    @classmethod
    @override
    def create(cls,data_params: dict[str, Any]) -> Loader:
        data_train = datasets.FashionMNIST(
            root=_DATASETS_ROOT_PATH,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        data_test = datasets.FashionMNIST(
            root=_DATASETS_ROOT_PATH,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return cls(data_train, data_test, data_params, "fashion-mnist")

class CifarLoader(Loader):
    @classmethod
    def create(cls, data_params: dict[str, Any]) -> Loader:
        data_train = datasets.CIFAR10(
            root=_DATASETS_ROOT_PATH,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        data_test = datasets.CIFAR10(
            root=_DATASETS_ROOT_PATH,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        return cls(data_train, data_test, data_params, "cifar-10")
