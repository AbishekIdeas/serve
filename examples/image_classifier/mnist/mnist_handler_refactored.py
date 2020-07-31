import io
import logging
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class MNISTDigitClassifier(BaseHandler):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        super(MNISTDigitClassifier, self).__init__()

    def load_model_path(self,ctx):
        """
        This prepares the model path. It is overriden from the BaseHandler 
        for demonstration purposes. This can be omitted if you are not 
        adding additional functionalities
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        self.model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        self.model_pt_path = os.path.join(self.model_dir, serialized_file)
        if not os.path.isfile(self.model_pt_path):
            raise RuntimeError("Missing the model.pt file")

    def load_label_mapping(self):
        """
        This loads the labels from a json file provided. It is overriden 
        from the BaseHandler for demonstration purposes.
        """
        mapping_file_path = os.path.join(self.model_dir, "index_to_name.json")

        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning('Missing the index_to_name.json file. Inference output will not include class name.')

    def get_input(self,data):
        """
        This gets the data from request and pass it to the preprocess function.
        It is overriden from the BaseHandler for demonstration purposes.
        """
        text = data[0].get("data")
        if text is None:
            input_data = data[0].get("body")
        return input_data

    def preprocess(self, image):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array
        """
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = Image.open(io.BytesIO(image))
        image = mnist_transform(image)
        return image

    def inference(self, img, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)

        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return [predicted_idx]

    def postprocess(self, inference_output):
        return inference_output

handle = MNISTDigitClassifier().get_default_handler()
