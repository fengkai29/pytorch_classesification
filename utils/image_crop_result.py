from torchvision import datasets, models, transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

test_transforms = transforms.Compose([
                # transforms.RandomRotation(degrees=15),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomGrayscale(),
                # transforms.ColorJitter(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                #transforms.RandomResizedCrop(224, scale=(0.49, 1.0)),
                # transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # transforms.ToPILImage(),

            ])

# path = '/Users/fengkai/PycharmProjects/pytorch_test/pytorch_classification/dataset/test_new/images'
# img_file = os.listdir(path)
# image = ['001omJHwzy6LA1I89Ds92&690.jpg', '0027.jpg', '031.jpg', '065.jpg', '08050343524181694.jpg', '101.jpg', '111.jpg', '123.jpg', '132.jpg', '147253756530297635.jpg', '1c1150850d434f998db4aa77806d8610.jpg', '2.jpg', '20151118180238_58930.jpg', '22.jpg', '27.jpg', '30.jpg', '32.jpg', '37.jpg', '38.jpg', '4010.jpg', '404.jpg', '408.jpg', '409.jpg', '41.jpg', '43.jpg', '46.jpg', '47.jpg', '5-160PPT005-50.jpg', '5-160PPT007-50.jpg', '52.jpg', '53.jpg', '57.jpg', '60.jpg', '67.jpg', '6b8dc1aed6e941beab7aed0f3169e14a.jpg', '7.jpg', '81.jpg', '92.jpg', '9cf272e8e8058657564d37ad3af3501469893c46.jpg', 'b2de9c82d158ccbf9f51297b14d8bc3eb0354186.jpg', 'CqgNOlhDWWeAXsqkAAAAAAAAAAA416.960x1201.jpg', 'dc54564e9258d1091111611fd758ccbf6d814da9.jpg', 'guangzhuobangzidaimojingdenanzishiliangsucai_4004144.jpg', 't01fb5cb9d6526dc145.jpg', 'u=1368371485,4131172081&fm=27&gp=0.jpg', 'u=1393875421,2137834719&fm=27&gp=0.jpg', 'u=1915364374,830347313&fm=27&gp=0.jpg', 'u=2019711001,1893601070&fm=26&gp=0.jpg', 'u=2258736311,2409298767&fm=27&gp=0.jpg', 'u=2499523155,4195216542&fm=15&gp=0.jpg', 'u=2502597282,145060923&fm=26&gp=0.jpg', 'u=2909630911,2182462009&fm=26&gp=0.jpg', 'u=3183275218,3014654585&fm=27&gp=0.jpg', 'u=3981980683,3495465359&fm=26&gp=0.jpg', 'u=633379299,454064171&fm=15&gp=0.jpg', 'W020131201524301127014.jpg', 'W020140605384135478444.jpg', 'W020140605384138902971.jpg', 'W020140605384143593986.jpg']

img_convert = 'ppt.jpg'
def pil_loader(imgpath):
    with open(imgpath, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transforms = test_transforms
# for i in img_file:
#     if i in image:
#         img = pil_loader(os.path.join(path,i))
#         if transforms is not None:
#             img = transforms(img)
#         plt.figure("Image")
#         plt.imshow(img)
#         plt.axis('on')
#         plt.title('image')
#         plt.show()

img = pil_loader(img_convert)
if transforms is not None:
    img = transforms(img)
plt.figure("Image")
plt.imshow(img)
plt.axis('on')
plt.title('image')
plt.show()