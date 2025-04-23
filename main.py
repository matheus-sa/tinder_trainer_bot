# main.py
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
import logging
import csv
import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import joblib

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = "feedback.csv"
MODEL_FILE = "model.pkl"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file_id", "decision", "model_prediction", "model_confidence"])


# Funzione di feature extraction semplificata
def extract_features(file_id):
    h = hashlib.sha256(file_id.encode()).digest()
    vec = np.frombuffer(h, dtype=np.uint8)[:16]
    return vec.astype(np.float32) / 255.0


# Predizione reale
def predict_with_model(features, data_file):
    try:
        df = pd.read_csv(data_file)
        X = np.array(
            [extract_features(row['file_id']) for _, row in df.iterrows()])
        y = np.array([
            1 if row['decision'] == 'like' else 0 for _, row in df.iterrows()
        ])
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        proba = model.predict_proba([features])[0][1]
        return ('like' if proba >= 0.5 else 'dislike', round(proba, 2))
    except Exception as e:
        print("Model prediction failed:", e)
        return ("like", 0.5)


# /info: Mostra i parametri del modello
def info(update: Update, context: CallbackContext):
    try:
        model = joblib.load(MODEL_FILE)
        coef = model.coef_[0]
        weights = "\n".join(
            [f"Feature {i+1}: {round(w, 3)}" for i, w in enumerate(coef)])
        update.message.reply_text(f"📐 Pesi attuali del modello:\n{weights}")
    except Exception as e:
        update.message.reply_text(f"❌ Modello non ancora disponibile: {e}")


# /nuova <nome>
def nuova(update: Update, context: CallbackContext):
    if context.args:
        nome = context.args[0]
        context.user_data['session_nome'] = nome
        context.user_data['session_photos'] = []
        update.message.reply_text(
            f"Inizia una nuova sessione per: {nome}. Inviami le foto.")
    else:
        update.message.reply_text("Usa /nuova <nome_ragazza>")


# /valuta
def valuta(update: Update, context: CallbackContext):
    nome = context.user_data.get('session_nome')
    photos = context.user_data.get('session_photos', [])
    if not nome or not photos:
        update.message.reply_text(
            "Devi prima usare /nuova e inviarmi delle foto.")
        return
    features = np.array([extract_features(f) for f in photos])
    avg_feat = features.mean(axis=0)
    prediction, confidence = predict_with_model(avg_feat, DATA_FILE)
    context.user_data['last_photo'] = nome + "_multi"
    context.user_data['last_prediction'] = prediction
    context.user_data['last_confidence'] = confidence
    context.user_data['photo_count'] = len(photos)
    context.user_data['multi_session'] = True
    keyboard = [[
        InlineKeyboardButton("👍 Mi piace", callback_data='like'),
        InlineKeyboardButton("👎 Non mi piace", callback_data='dislike')
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text(
        f"📊 Predizione media per {nome}: {'❤️' if prediction == 'like' else '❌'} ({confidence * 100:.0f}%)\nTi piace questa ragazza?",
        reply_markup=reply_markup)


# Funzione /gusti
def gusti(update: Update, context: CallbackContext):
    try:
        df = pd.read_csv(DATA_FILE)
        df_like = df[df['decision'] == 'like']
        if df_like.empty:
            update.message.reply_text(
                "Non ho abbastanza dati sui tuoi like per fare un'analisi.")
            return
        update.message.reply_text(
            f"Hai messo ❤️ a {len(df_like)} ragazze. Stai costruendo un modello estetico coerente!"
        )
    except Exception as e:
        update.message.reply_text(f"Errore: {e}")


# Funzione /start
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Ciao! Inviami una foto o inizia una sessione con /nuova <nome> per valutarne più di una."
    )


# Funzione /train
def train(update: Update, context: CallbackContext):
    try:
        df = pd.read_csv(DATA_FILE)
        X = np.array(
            [extract_features(row['file_id']) for _, row in df.iterrows()])
        y = np.array([
            1 if row['decision'] == 'like' else 0 for _, row in df.iterrows()
        ])
        model = LogisticRegression()
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        update.message.reply_text("✅ Modello riaddestrato con successo su " +
                                  str(len(X)) + " esempi.")
    except Exception as e:
        update.message.reply_text("❌ Errore durante l'addestramento: " +
                                  str(e))


# Funzione /stats
def stats(update: Update, context: CallbackContext):
    try:
        df = pd.read_csv(DATA_FILE)
        total = len(df)
        likes = df[df['decision'] == 'like'].shape[0]
        dislikes = total - likes
        correct_preds = df[df['decision'] == df['model_prediction']].shape[0]
        accuracy = (correct_preds / total) * 100 if total else 0
        avg_conf = df['model_confidence'].astype(float).mean()
        update.message.reply_text(f"📊 Statistiche raccolta:\n"
                                  f"Totale voti: {total}\n"
                                  f"❤️ Mi piace: {likes}\n"
                                  f"❌ Non mi piace: {dislikes}\n"
                                  f"✅ Accuratezza modello: {accuracy:.1f}%\n"
                                  f"📈 Confidenza media: {avg_conf:.2f}")
    except Exception as e:
        update.message.reply_text("❌ Errore nel calcolo delle statistiche: " +
                                  str(e))


# Gestione foto
def handle_photo(update: Update, context: CallbackContext):
    file_id = update.message.photo[-1].file_id
    if 'session_photos' in context.user_data:
        context.user_data['session_photos'].append(file_id)
        update.message.reply_text(
            f"Foto aggiunta alla sessione. Totale: {len(context.user_data['session_photos'])}"
        )
        return
    context.user_data['last_photo'] = file_id
    features = extract_features(file_id)
    prediction, confidence = predict_with_model(features, DATA_FILE)
    context.user_data['last_prediction'] = prediction
    context.user_data['last_confidence'] = confidence
    text = f"📊 Il modello pensa che: {'❤️ Ti piace' if prediction == 'like' else '❌ Non ti piace'} ({confidence * 100:.0f}%)"
    keyboard = [[
        InlineKeyboardButton("👍 Mi piace", callback_data='like'),
        InlineKeyboardButton("👎 Non mi piace", callback_data='dislike')
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo=file_id,
                           caption=text + "\nTi piace questa ragazza?",
                           reply_markup=reply_markup)


# Callback dei bottoni
def button(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()
    choice = query.data
    file_id = context.user_data.get('last_photo')
    prediction = context.user_data.get('last_prediction')
    confidence = context.user_data.get('last_confidence')
    if file_id:
        with open(DATA_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([file_id, choice, prediction, confidence])
    try:
        query.edit_message_caption(caption=f"Hai scelto: {'❤️' if choice ==             'like' else '❌'}")
    except:
        query.edit_message_text(text=f"Hai scelto: {'❤️' if choice == 'like'            else '❌'}")

    if context.user_data.get('multi_session'):
        context.user_data.pop('session_photos', None)
        context.user_data.pop('session_nome', None)
        context.user_data.pop('multi_session', None)
        query.message.reply_text("✔️ Sessione terminata.")


# Main
if __name__ == '__main__':
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("nuova", nuova))
    dp.add_handler(CommandHandler("valuta", valuta))
    dp.add_handler(CommandHandler("gusti", gusti))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(CommandHandler("stats", stats))
    dp.add_handler(CommandHandler("info", info))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    dp.add_handler(CallbackQueryHandler(button))
    updater.start_polling()
    updater.idle()
