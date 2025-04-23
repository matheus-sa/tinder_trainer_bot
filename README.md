# Photo Feedback Telegram Bot

A Telegram bot that collects user feedback (like/dislike) on photos and stores results in a CSV file.

## Features

- Accepts photo submissions from users
- Provides interactive like/dislike voting system using inline buttons
- Stores user feedback in CSV format with timestamps
- Basic command handling (/start, /help)
- Photo display with caption asking for user opinion
- Confirmation messages after voting

## Setup Instructions

1. **Get a Telegram Bot Token**
   - Talk to the [BotFather](https://t.me/botfather) on Telegram
   - Send the `/newbot` command and follow the instructions
   - Copy the API token provided

2. **Set Environment Variables**
   ```
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   ```
   
   Alternatively, you can directly edit the token in the script, but this is not recommended for security reasons.

3. **Install Dependencies**
   ```
   pip install python-telegram-bot
   ```

4. **Run the Bot**
   ```
   python main.py
   ```

## Usage

1. Start a chat with your bot on Telegram
2. Send the `/start` command to get a welcome message
3. Send a photo to the bot
4. The bot will display the photo and ask if you like it
5. Click the 👍 or 👎 button to provide your feedback
6. Your feedback will be stored in the `feedback.csv` file

## Data Storage

The bot stores feedback data in a CSV file named `feedback.csv` with the following columns:
- `file_id`: Telegram's unique identifier for the photo
- `decision`: User's choice ('like' or 'dislike')
- `timestamp`: When the feedback was provided
- `user_id`: Telegram user ID of the person providing feedback

## Commands

- `/start` - Start the bot and get a welcome message
- `/help` - Display help information about using the bot

## Security Notes

- Never share your bot token publicly
- The bot stores minimal user data (just Telegram's user ID)
- No personal information is collected beyond what Telegram provides

## License

This project is open source and available under the MIT License.
