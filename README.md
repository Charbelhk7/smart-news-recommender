# Smart News Recommender

A personalized news video recommender system designed for short-form and long-form videos, built as part of the MSBA Capstone Project at the **American University of Beirut (AUB)**.

This system mimics the user experience of platforms like TikTok and YouTube Shorts, delivering regionally relevant and engaging news content based on user preferences.

---

## 🎯 Project Purpose

The app recommends news videos based on:

- 🎭 **Tone**
- 🗣️ **Style**
- 💬 **Emotion**
- 🌍 **Region**
- ⏱️ **Duration**
- 🌐 **Language**
- 📰 **Topic**

---

## 🛠️ Technologies Used

- `Python` (pandas, scikit-learn, joblib)
- `Streamlit` (frontend + backend)
- `OpenAI GPT-4o` for feature extraction
- `XGBoost` for training the recommendation model
- `GitHub` for version control
- `Streamlit Cloud` for deployment

---

## 📂 Files in This Repository

| File | Description |
|------|-------------|
| `final_app.py` | The Streamlit frontend + recommendation logic |
| `final_model.joblib` | The trained XGBoost recommendation model |
| `feature_columns.pkl` | Feature column structure used during training |
| `videos_ready_for_matching.csv` | Cleaned and tagged video metadata |
| `requirements.txt` | Python dependencies for deployment |

---

## 🚀 Live App

Access the live app here:  
👉 [Smart News Recommender](https://smart-news-recommender-pzbh8mjxwqsuf3yac9xryv.streamlit.app)

---

## 📍 About the Project

This system was developed using 819 real video transcripts from a MENA-based multilingual news platform. It demonstrates a content-based approach to solving the cold-start problem in recommender systems.

---

## 👨‍🎓 Built for

**MSBA Capstone Project – Spring 2025**  
**American University of Beirut (AUB)**

---

## 📬 Contact

For academic or professional inquiries, contact the project author via GitHub.
