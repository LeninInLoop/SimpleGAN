from model import transforms

class Transformers:
   @staticmethod
   def get_mnist_transform():
       return transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
       ])
