from monai.utils import set_determinism
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureTyped,
    Invertd,
    Resized,
    SaveImaged,
)
from monai.data import DataLoader, Dataset
import os
import glob
set_determinism(seed=0)


outputDir="./output"


train_images = sorted(glob.glob(os.path.join( "./skullrec_dataset/train/cranial/", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join( "./skullrec_dataset/train/ground_truth/", "*.nii.gz")))
train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

test_images = sorted(glob.glob(os.path.join('./skullrec_dataset/test/cranial/', "*.nii.gz")))
test_label = sorted(glob.glob(os.path.join('./skullrec_dataset/test/ground_truth/', "*.nii.gz")))
test_files = [{"image": image} for image in test_images]


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"],spatial_size=(256, 256, 128),mode='nearest'),
        EnsureTyped(keys=["image", "label"]),
        #SaveImaged(keys=["image"],output_dir='./train_transformed_data')
        #ToDeviced(keys=["image", "label"],device='cuda:0'),
    ]
)

test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Resized(keys=["image"],spatial_size=(256, 256, 128),mode='nearest'),
        EnsureTyped(keys="image"),
        #SaveImaged(keys=["image"],output_dir='./train_transformed_data')
    ]
)




test_org_transforms1 = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
    ]
)

#os.mkdir('output')

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=test_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True),
    # Specify here the output directory. 
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=outputDir, output_postfix="completed", resample=False),
])



train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)


