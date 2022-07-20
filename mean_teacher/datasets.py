import torchvision.transforms as transforms

from .utils import export


@export
def raf():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    #DAN transformation:
    train_transformation =  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])

    train_transformation2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])



    eval_transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    return {
        'train_transformation': train_transformation,
        'train_transformation2': train_transformation2,
        'eval_transformation': eval_transformation,
        'datadir': '/mnt/Data/tohar/raf-basic/basic/',
        'num_classes': 7
    }


