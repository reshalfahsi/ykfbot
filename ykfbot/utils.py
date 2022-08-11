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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import subprocess
import os
import re

from .constant import VOCAB_SIZE, MAX_LENGTH


def remove_dir(d):
    if os.name == "nt":
        subprocess.check_output(["cmd", "/C", "rmdir", "/S", "/Q", os.path.abspath(d)])
    else:
        subprocess.check_output(["rm", "-rf", os.path.abspath(d)])


def load_conversations(path_to_movie_lines, path_to_movie_conversations):
    # Helper function for loading the conversation splits
    id2line = {}
    with open(path_to_movie_lines, errors="ignore") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        id2line[parts[0]] = parts[4]

    inputs, outputs = [], []
    with open(path_to_movie_conversations, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.replace("\n", "").split(" +++$+++ ")
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(", ")]
        for i in range(len(conversation) - 1):
            inputs.append(id2line[conversation[i]])
            outputs.append(id2line[conversation[i + 1]])
    return inputs, outputs


def preprocess_text(sentence):
    sentence = tf.strings.lower(sentence)
    # Adding a space between the punctuation and the last word to allow better tokenization
    sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")
    # Replacing multiple continuous spaces with a single space
    sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
    # Replacing non english words with spaces
    sentence = tf.strings.regex_replace(sentence, r"[^a-z?.!,]+", " ")
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence


def get_vectorizer():
    path_to_zip = keras.utils.get_file(
        "cornell_movie_dialogs.zip",
        origin="http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        extract=True,
    )

    path_to_dataset = os.path.join(
        os.path.dirname(path_to_zip), "cornell movie-dialogs corpus"
    )

    path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
    path_to_movie_conversations = os.path.join(
        path_to_dataset, "movie_conversations.txt"
    )

    questions, answers = load_conversations(path_to_movie_lines, path_to_movie_conversations)

    vectorizer = layers.TextVectorization(
        VOCAB_SIZE,
        standardize=preprocess_text,
        output_mode="int",
        output_sequence_length=MAX_LENGTH,
    )
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices((questions + answers)).batch(128)
    )

    remove_dir(os.path.dirname(path_to_zip))

    return vectorizer
