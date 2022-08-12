# MIT License
#
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified from:
#    - https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_generation_fnet.py
# ==============================================================================

from .model import create_model
from .utils import get_vectorizer, preprocess_text
from .constant import MODEL_URL, MAX_LENGTH

import urllib.request
import tensorflow as tf

from random_word import RandomWords


class YourKindFriendBot(object):
    def __init__(self):
        self.vectorizer = get_vectorizer()

        urllib.request.urlretrieve(MODEL_URL, "ykfbot.h5")

        self.model = create_model()
        self.model.load_weights("ykfbot.h5")

        self.r = RandomWords()

    def reply(self, input_sentence):

        VOCAB = self.vectorizer.get_vocabulary()

        # Mapping the input sentence to tokens and adding start and end tokens
        tokenized_input_sentence = self.vectorizer(
            tf.constant("[start] " + preprocess_text(input_sentence) + " [end]")
        )
        # Initializing the initial sentence consisting of only the start token.
        tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
        decoded_sentence = ""
        new_sentence = True
        apostrophe = ["t", "re", "ll", "m"]
        punctuation = [".", "?", "!"]

        for i in range(MAX_LENGTH):
            # Get the predictions
            predictions = self.model.predict(
                {
                    "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
                    "decoder_inputs": tf.expand_dims(
                        tf.pad(
                            tokenized_target_sentence,
                            [[0, MAX_LENGTH - tf.shape(tokenized_target_sentence)[0]]],
                        ),
                        0,
                    ),
                }
            )
            # Calculating the token with maximum probability and getting the corresponding word
            sampled_token_index = tf.argmax(predictions[0, i, :])
            sampled_token = VOCAB[sampled_token_index.numpy()]
            # If sampled token is the end token then stop generating and return the sentence
            if tf.equal(sampled_token_index, VOCAB.index("[end]")):
                break

            if tf.equal(sampled_token_index, VOCAB.index("[UNK]")):
                sampled_token = self.r.get_random_word(includePartOfSpeech="noun")

            if i == 0 or new_sentence or sampled_token == "i":
                sampled_token = sampled_token.title()
                new_sentence = False

            if sampled_token in punctuation:
                decoded_sentence = decoded_sentence[:-1]
                new_sentence = True

            if sampled_token == ",":
                decoded_sentence = decoded_sentence[:-1]

            if sampled_token in apostrophe:
                decoded_sentence = decoded_sentence[:-1] + "'"

            decoded_sentence += sampled_token + " "
            tokenized_target_sentence = tf.concat(
                [tokenized_target_sentence, [sampled_token_index]], 0
            )

        return decoded_sentence[:-1]
