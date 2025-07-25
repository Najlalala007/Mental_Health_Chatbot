{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e4633a19-3914-4f3d-b679-dd8316631c4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Matplotlib created a temporary cache directory at C:\\Users\\nazla\\AppData\\Local\\Temp\\matplotlib-vpd4coys because the default path (C:\\Users\\nazla\\.matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.\n",
      "Matplotlib is building the font cache; this may take a moment.\n",
      "2025-01-31 15:02:38.006 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\nazla\\Documents\\Anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "C:\\Users\\nazla\\Documents\\Anaconda\\Lib\\site-packages\\sklearn\\base.py:318: UserWarning: Trying to unpickle estimator LabelEncoder from version 1.6.1 when using version 1.2.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n",
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import pickle\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "\n",
    "# Load Tokenizer and Label Encoders\n",
    "@st.cache_resource\n",
    "def load_resources():\n",
    "    try:\n",
    "        with open(\"tokenizer.pkl\", \"rb\") as f:\n",
    "            tokenizer = pickle.load(f)\n",
    "\n",
    "        with open(\"sentiment_label_encoder.pkl\", \"rb\") as f:\n",
    "            sentiment_label_encoder = pickle.load(f)\n",
    "\n",
    "        with open(\"subreddit_label_encoder.pkl\", \"rb\") as f:\n",
    "            subreddit_label_encoder = pickle.load(f)\n",
    "\n",
    "        # Load trained models\n",
    "        sentiment_model = tf.keras.models.load_model(\"sentiment_lstm_model.h5\")\n",
    "        subreddit_model = tf.keras.models.load_model(\"subreddit_lstm_model.h5\")\n",
    "\n",
    "        return tokenizer, sentiment_label_encoder, subreddit_label_encoder, sentiment_model, subreddit_model\n",
    "    except Exception as e:\n",
    "        st.error(f\"Error loading models or files: {e}\")\n",
    "        st.stop()\n",
    "\n",
    "# Load resources\n",
    "tokenizer, sentiment_label_encoder, subreddit_label_encoder, sentiment_model, subreddit_model = load_resources()\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"🧠 Mental Health & Sentiment Analysis App\")\n",
    "st.write(\"Enter text below to analyze both **sentiment** and **mental health category**.\")\n",
    "\n",
    "user_input = st.text_area(\"Enter your text here:\")\n",
    "\n",
    "if st.button(\"Analyze\"):\n",
    "    if user_input.strip():\n",
    "        try:\n",
    "            # Preprocess input text\n",
    "            sequence = tokenizer.texts_to_sequences([user_input])\n",
    "            padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')\n",
    "\n",
    "            # Sentiment Prediction\n",
    "            sentiment_pred = sentiment_model.predict(padded_sequence)\n",
    "            sentiment_label = sentiment_label_encoder.inverse_transform([np.argmax(sentiment_pred)])[0]\n",
    "\n",
    "            # Mental Health Prediction\n",
    "            subreddit_pred = subreddit_model.predict(padded_sequence)\n",
    "            subreddit_label = subreddit_label_encoder.inverse_transform([np.argmax(subreddit_pred)])[0]\n",
    "\n",
    "            # Display results\n",
    "            st.subheader(\"🔍 Analysis Results:\")\n",
    "            st.write(f\"**📌 Sentiment:** {sentiment_label}\")\n",
    "            st.write(f\"**💭 Mental Health Category:** {subreddit_label}\")\n",
    "\n",
    "        except Exception as e:\n",
    "            st.error(f\"An error occurred during prediction: {e}\")\n",
    "\n",
    "    else:\n",
    "        st.warning(\"⚠️ Please enter some text to analyze.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    st.write(\"🚀 Ready for analysis!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e66c9634-9507-474d-87c1-ab8f907e7f2d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
