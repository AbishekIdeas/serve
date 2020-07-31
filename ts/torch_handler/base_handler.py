"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import abc
import logging
import os
import json
import importlib

import torch
from ..utils.util import list_classes_from_module
logger = logging.getLogger(__name__)

_singleton_handler  = None

class BaseHandler(abc.ABC):
    """
    Base default handler to load torchscript or eager mode [state_dict] models
    Also, provides handle method per torch serve custom model specification
    """
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.manifest = None
        self.model_dir = None
        self.model_pt_path = None

    def load_model_path(self,ctx):
        """
        This prepares the model path. 
        This can be overriden in the CustomHandlers
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
        This loads the labels from a json file provided.
        This can be overriden in the CustomHandlers
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
        This can be overriden in the CustomHandlers
        """
        text = data[0].get("data")
        if text is None:
            input_data = data[0].get("body")
        return input_data

    def map_label_to_class(output):
        
        if self.mapping:
            output = self.mapping[str(output)]

    def initialize(self, ctx):
        """
        First try to load torchscript else load eager mode state_dict based model
        """
        self.load_model_path(ctx)

        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'modelFile' in self.manifest['model']:
            model_file = self.manifest['model']['modelFile']
            module = importlib.import_module(model_file.split(".")[0])
            model_class_definitions = list_classes_from_module(module)
            if len(model_class_definitions) != 1:
                raise ValueError("Expected only one class as model definition. {}".format(
                    model_class_definitions))

            model_class = model_class_definitions[0]
            state_dict = torch.load(self.model_pt_path, map_location=map_location)
            self.model = model_class()
            self.model.load_state_dict(state_dict)
        else:
            logger.debug('No model file found for eager mode, trying to load torchscript model')
            self.model = torch.jit.load(self.model_pt_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file %s loaded successfully', self.model_pt_path)
        # Read the mapping file, index to object name
        self.load_label_mapping()

        self.initialized = True

    def preprocess(self, input_data):
        """
        The default implementation of preprocess. This will get executed
        if the user doesn't provide implementation in his CustomHandler
        """
        input_data = torch.as_tensor(input_data)
        return [input_data]

    def inference(self, data):
        """
        The default implementation of inference. This will get executed
        if the user doesn't provide implementation in his CustomHandler
        """
        torch_data = torch.as_tensor(data, device=self.device)
        output = self.model(torch_data)
        output = output.to('cpu')
        return [output]

    def postprocess(self, data):
        """
        The default implementation of postprocess. This will get executed
        if the user doesn't provide implementation in his CustomHandler
        """
        return data
    

    def _handle(self, data, context):
        """
        Entry point for CustomHandlers and Default Handlers. 
        """
        try :
            if data is None:
                return None

            input_data = self.get_input(data)
            processed = self.preprocess(input_data)
            predictions = self.inference(processed)
            output = self.postprocess(predictions)

        except Exception as e:
            raise Exception("Please provide a custom handler in the model archive." + e)

        return output
     
    @classmethod
    def get_default_handler(cls, *args, **kwargs):
        """
        This provides a singleton handler to load the model on first request alone.
        This should be called from the Custom and default handlers to get the handle function.
        """
        def handle(data, context):
            global _singleton_handler
            if _singleton_handler is None:
                _singleton_handler = cls(*args, **kwargs)
            if not _singleton_handler.initialized:
                _singleton_handler.initialize(context)
            return _singleton_handler._handle(data, context)
        return handle
    


    
