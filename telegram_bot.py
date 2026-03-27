from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from auth import update_chat_id

TOKEN = "YOUR_BOT_TOKEN"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    chat_id = update.effective_chat.id

    if context.args:
        email = context.args[0]

        update_chat_id(email, chat_id)

        await update.message.reply_text(
            "✅ Telegram notifications connected successfully!"
        )

    else:
        await update.message.reply_text(
            "Send the connect link from the website."
        )

app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))

app.run_polling()