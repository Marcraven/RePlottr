import os
import requests
from datetime import datetime
from donutplottr.params import *


def bot_notify(old_mae, new_mae):
    """
    Notify about the performance
    """

    TOKEN = "6675096823:AAFsa0GRiYtUKp1o1uqTFNfyA7NQrq725Vo"
    chat_id = "727995087"

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
