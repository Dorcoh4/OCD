import torch
import torch.nn as nn
import numpy as np
from nerf_utils.nerf import cumprod_exclusive, get_minibatches, get_ray_bundle, positional_encoding
from nerf_utils.tiny_nerf import VeryTinyNerfModel
from torchvision.datasets import mnist
from torchvision import transforms
import Lenet5
import bert_ocd
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from copy import deepcopy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import datasets

def wrapper_dataset(config, args, device):
    if args.datatype == 'tinynerf':
        
        data =  np.load(args.data_train_path)
        images = data["images"]
        # Camera extrinsics (poses)
        tform_cam2world = data["poses"]
        tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)

        # Height and width of each image
        height, width = images.shape[1:3]

        # Near and far clipping thresholds for depth values.
        near_thresh = 2.0
        far_thresh = 6.0

        # Hold one image out (for test).
        testimg, testpose = images[101], tform_cam2world[101]
        testimg = torch.from_numpy(testimg).to(device)

        # Map images to device
        images = torch.from_numpy(images[:100, ..., :3]).to(device)
        num_encoding_functions = 10
        # Specify encoding function.
        encode = positional_encoding
        # Number of depth samples along each ray.
        depth_samples_per_ray = 32
        model = VeryTinyNerfModel(num_encoding_functions=num_encoding_functions)
        # Chunksize (Note: this isn't batchsize in the conventional sense. This only
        # specifies the number of rays to be queried in one go. Backprop still happens
        # only after all rays from the current "bundle" are queried and rendered).
        # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory (when using 8
        # samples per ray).
        chunksize = 4096
        batch = {}
        batch['height'] = height
        batch['width'] = width
        batch['focal_length'] = focal_length
        batch['testpose'] = testpose
        batch['near_thresh'] = near_thresh
        batch['far_thresh'] = far_thresh
        batch['depth_samples_per_ray'] = depth_samples_per_ray
        batch['encode'] = encode
        batch['get_minibatches'] =get_minibatches
        batch['chunksize'] =chunksize
        batch['num_encoding_functions'] = num_encoding_functions
        train_ds, test_ds = [],[]
        for img,tfrom in zip(images,tform_cam2world):
            batch['input'] = tfrom
            batch['output'] = img
            train_ds.append(deepcopy(batch))
        batch['input'] = testpose
        batch['output'] = testimg
        test_ds = [batch]
    elif args.datatype == 'mnist':
        model = Lenet5.NetOriginal()
        train_transform = transforms.Compose(
                            [
                            transforms.ToTensor()
                            ])
        train_dataset = mnist.MNIST(
                "\data\mnist", train=True, download=True, transform=ToTensor())
        test_dataset = mnist.MNIST(
                "\data\mnist", train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [],[]
        for idx, data in enumerate(train_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            train_x, train_label = data[0], data[1]
            train_x = train_x[:,0,:,:].unsqueeze(1)
            batch = {'input':train_x,'output':train_label}
            test_ds.append(deepcopy(batch))
    elif args.datatype == 'scicite':
        model = bert_ocd.BertOriginal(AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).cuda())
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        ds = datasets.load_dataset("scicite")

        # train_dataset = mnist.MNIST(
        #         "\data\mnist", train=True, download=True, transform=ToTensor())
        # test_dataset = mnist.MNIST(
        #         "\data\mnist", train=False, download=True, transform=ToTensor())
        def tokenize_dataset(data):
            # Keys of the returned dictionary will be added to the dataset as columns
            return tokenizer(data["string"], padding="max_length", truncation=True)

        tokenized_ds = ds.map(tokenize_dataset)
        tokenized_ds = tokenized_ds.remove_columns(["string"])
        tokenized_ds = tokenized_ds.rename_column("label", "labels")
        tokenized_ds.set_format("torch")
        train_dataset = tokenized_ds['train']
        test_dataset = tokenized_ds['test']
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1)
        train_ds, test_ds = [], []
        for idx, data in enumerate(train_loader):
            train_x, train_label = data['input_ids'], data['labels']
            # train_x = train_x[:, 0].unsqueeze(1)
            batch = {'input': train_x, 'attention_mask': data['attention_mask'], 'output': train_label}
            train_ds.append(deepcopy(batch))
        for idx, data in enumerate(test_loader):
            test_x, test_label = data['input_ids'], data['labels']
            # test_x = test_x[:, 0].unsqueeze(1)
            batch = {'input': test_x, 'attention_mask': data['attention_mask'], 'output': test_label}
            test_ds.append(deepcopy(batch))
    else:
        "implement on your own"
        pass
    return train_ds,test_ds,model
