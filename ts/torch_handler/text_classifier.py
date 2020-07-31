# pylint: disable=E1102
# TODO remove pylint disable comment after https://github.com/pytorch/pytorch/issues/24807 gets merged.
"""
Module for text classification default handler
"""
import torch
from torch.autograd import Variable
from torchtext.data.utils import ngrams_iterator
from .text_handler import TextHandler


class TextClassifier(TextHandler):
    """
    TextClassifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the model vocabulary.
    """

    def __init__(self):
        super(TextClassifier, self).__init__()

    def preprocess(self, text):
        """
        Normalizes the input text for PyTorch model using following basic cleanup operations :
            - remove html tags
            - lowercase all text
            - expand contractions [like I'd -> I would, don't -> do not]
            - remove accented characters
            - remove punctuations
        Converts the normalized text to tensor using the source_vocab.
        Returns a Numpy array.
        """
        ngrams = 2
        text = text.decode('utf-8')

        text = self._remove_html_tags(text)
        text = text.lower()
        text = self._expand_contractions(text)
        text = self._remove_accented_characters(text)
        text = self._remove_puncutation(text)
        text = self._tokenize(text)
        text = torch.tensor([self.source_vocab[token] for token in ngrams_iterator(text, ngrams)])

        return text

    def inference(self, data):
        """
        Predict the class of a text using a trained deep learning model and vocabulary.
        """

        inputs = Variable(data).to(self.device)
        output = self.model.forward(inputs, torch.tensor([0]).to(self.device))
        output = output.argmax(1).item() + 1
        if self.mapping:
            output = self.mapping[str(output)]

        return [output]

    def postprocess(self, data):
        return data


handle = TextClassifier().get_default_handler()
