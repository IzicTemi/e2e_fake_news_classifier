# pylint: disable=import-error
# pylint: disable=no-name-in-module

from tensorflow.keras.preprocessing import sequence


class ModelServer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def prepare_tokens(self, text):
        maxlen = 300
        tokenizer = self.tokenizer
        tokens = tokenizer.texts_to_sequences([text])
        prepped_tokens = sequence.pad_sequences(tokens, maxlen=maxlen)
        return prepped_tokens

    def classify(self, prepped_tokens):
        print("Loading the model...")
        model = self.model
        print("Successful!")
        preds = model.predict(prepped_tokens)
        return preds

    def lambda_handler(self, event):
        text = event['text']

        tokens = self.prepare_tokens(text)

        pred = self.classify(tokens)

        int_pred = (pred > 0.5).astype("int32").tolist()[0][0]

        dict_map = {
            0: False,
            1: True,
        }

        final_pred = dict_map[int_pred]

        result = {
            'text': text,
            'class': final_pred,
        }

        return result
