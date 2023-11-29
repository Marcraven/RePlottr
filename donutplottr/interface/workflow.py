import os
import requests


def bot_notify(old_mae, new_mae):
    """
    Notify about the performance
    """

    TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    if new_mae < old_mae and new_mae < 2.5:
        message = f"🚀 New model replacing old in production with MAE: {new_mae} the Old MAE was: {old_mae}"

    elif old_mae < 2.5:
        message = (
            f"✅ Old model still good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
        )
    else:
        message = f"🚨 No model good enough: Old MAE: {old_mae} - New MAE: {new_mae}"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"

    print(requests.get(url).json())
