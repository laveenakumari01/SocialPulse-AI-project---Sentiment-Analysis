# SocialPulse-AI-project---Sentiment-Analysis

🔍 What is SocialPulse AI?
SocialPulse AI is an end-to-end sentiment analysis web app built with Python and Streamlit. It classifies any text as Positive or Negative with a confidence score, using a Logistic Regression model trained on the Sentiment140 dataset.
This project demonstrates a complete ML pipeline — from raw data to a live deployed web application.

🌐 Live Demo
👉 Try it live →



✨ Features

🔮 Instant Sentiment Analysis — Positive / Negative with confidence score
📋 Batch Mode — Analyze up to 10 texts at once
📊 Analytics Dashboard — Session stats, distribution bar, confidence breakdown
🕓 History Panel — Track all previous analyses in a session
⚡ Quick Examples — Try preset tweets from the sidebar
🌙 Dark Professional UI — Custom dark theme with color-coded results


🖼️ Screenshots
DashboardAnalyzeAnalyticsLive stats & recent historySingle + Batch analysisDistribution charts

🧠 How It Works
Raw Tweet
   ↓
Text Cleaning  →  Remove URLs, mentions, hashtags, special chars
   ↓
Lowercasing + Tokenization
   ↓
Porter Stemming  →  running → run, loved → love
   ↓
TF-IDF Vectorization  →  50,000 features, bigrams (1,2)
   ↓
Logistic Regression  →  Predict: Positive / Negative
   ↓
Confidence Score (%)

📊 Model Performance
MetricScoreTraining Accuracy82.0%Testing Accuracy80.1%Precision (Positive)0.79Recall (Positive)0.82F1-Score0.80Test Set Size320,000 tweets

🛠️ Tech Stack
LayerTechnologyLanguagePython 3.10+Web FrameworkStreamlitML Modelscikit-learn — Logistic RegressionNLPNLTK — Porter Stemmer, StopwordsVectorizerTF-IDF (50K features, ngram 1–2)DatasetSentiment140 (1.6M tweets)

🚀 Run Locally
1. Clone the repository
bashgit clone https://github.com/your-username/socialpulse-ai.git
cd socialpulse-ai
2. Install dependencies
bashpip install -r requirements.txt
3. Add model files
Place these two files in the root folder (train the model or download from releases):
sentiment_model.pkl
vectorizer.pkl
4. Run the app
bashstreamlit run app.py
Then open http://localhost:8501 in your browser.

📁 Project Structure
socialpulse-ai/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sentiment_model.pkl     # Trained Logistic Regression model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
└── README.md               # This file

🏋️ Training the Model
The model was trained in Google Colab using the Sentiment140 dataset.
python# Key training settings
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    sublinear_tf=True
)

model = LogisticRegression(
    C=4.0,
    max_iter=3000,
    solver='saga',
    n_jobs=-1
)
To retrain the model yourself, open the Colab notebook and run all cells. The .pkl files are saved automatically.

⚠️ Limitations

Binary only — classifies Positive or Negative (no neutral class yet)
English only — trained on English tweets; other languages may give poor results
Sarcasm — irony and sarcasm can sometimes fool the model
Context — short or ambiguous texts may have lower confidence


🔮 Future Improvements

 Neutral sentiment class (3-way classification)
 BERT / Transformer fine-tuning for higher accuracy
 CSV file upload for bulk analysis
 Twitter/X URL input — paste a link and analyze directly
 Emoji and slang support
 Multi-language sentiment detection
 Export results as PDF or CSV


📚 Dataset
Sentiment140 by Go, Bhayani & Huang (Stanford University, 2009)

1,600,000 tweets
Labels: 0 = Negative, 4 = Positive (remapped to 0/1)
Source: Kaggle


👩‍💻 Author
Laveena Kumari

GitHub: @laveenakumari01


📄 License
This project is licensed under the MIT License — feel free to use, modify, and share.

🌟 Support
If you found this project helpful, please consider giving it a ⭐ on GitHub — it means a lot!

Built with ❤️ using Python, scikit-learn & Streamlit
