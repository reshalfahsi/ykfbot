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
#    - https://github.com/liuhh02/python-telegram-bot-heroku/blob/master/bot.py
# ==============================================================================


import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
import requests
import numpy as np
import cv2
import uuid

from ykfbot import YourKindFriendBot


bot = YourKindFriendBot()


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text(
        "Your typical kind friend who talk nonsense just to kill time."
    )


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text("Just type or send an image, it will reply.")


def reply_text(update, context):
    """Reply the user message."""
    update.message.reply_text(bot.reply(update.message.text))


def reply_photo(update, context):
    logger.info("Receive image from user.")
    filename = "image-{}.jpg".format(uuid.uuid4())
    image = update.message.photo[-1].get_file()
    image = image.download(filename)
    files = {'file':  open(filename, 'rb')}
    logger.info("Send image to API.")
    response = requests.post("https://wpir-dnjf-8439.herokuapp.com/predict", files=files)
    logger.info("Process the response.")
    image = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    cv2.imwrite(filename, image)
    logger.info("Send result to user.")
    update.message.reply_photo(photo=open(filename, 'rb'), caption="I found something in this image.")
    os.remove(filename)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary

    PORT = int(os.environ.get("PORT", "5000"))
    TOKEN = os.environ.get("TOKEN")

    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, reply_text))
    dp.add_handler(MessageHandler(Filters.photo, reply_photo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_webhook(
        listen="0.0.0.0",
        port=int(PORT),
        url_path=TOKEN,
        webhook_url="https://ykfbot.herokuapp.com/" + TOKEN,
    )


if __name__ == "__main__":
    main()
