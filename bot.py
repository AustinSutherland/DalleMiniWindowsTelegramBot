
from telegram.ext import ApplicationBuilder, CallbackContext, CommandHandler
from dalle import DalleMini
from telegram import Update

runner = DalleMini()

with open('key.txt') as f:
    token = f.readline().strip()

async def showme(update: Update, context: CallbackContext.DEFAULT_TYPE):
    try:
        query = " ".join(update.message.text.split(" ")[1:])
        chat_id = update.effective_chat.id

        if (len(query) == 0):
            await context.bot.send_message(chat_id=chat_id, text="Please provide a query", reply_to_message_id=update.message.message_id)
            return

        if (len(query) > 75):
            await context.bot.send_message(chat_id=chat_id, text="Query too long. Max 75 chars", reply_to_message_id=update.message.message_id)
            return

        await context.bot.send_message(chat_id=chat_id, text="Ok I am generating now, please wait a little bit !", reply_to_message_id=update.message.message_id)
        runner.generate(query)
        await context.bot.send_photo(chat_id, open('result.jpg', 'rb'), caption="Here is your image :)", reply_to_message_id=update.message.message_id)
    except:
        await context.bot.send_message(chat_id=chat_id, text="Something went wrong sorry :(")

def main():

    application = ApplicationBuilder().token(token).build()

    showme_handler = CommandHandler('showme', showme)
    application.add_handler(showme_handler)

    application.run_polling()


if __name__ == '__main__':
    main()
