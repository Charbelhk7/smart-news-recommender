# ---------------------------
# Step 1: Imports and File Loads
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from random import shuffle

st.set_page_config(page_title="Smart News Recommender", layout="centered")

st.markdown("""
    <style>
        .stButton > button {
            background-color: #0051a2;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0074d9;
            color: white;
        }
        .st-multiselect div[role="listbox"] {
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

model = joblib.load("final_model.joblib")

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

videos_df = pd.read_csv("videos_ready_for_matching.csv")

if "last_results" not in st.session_state:
    st.session_state.last_results = []

# ---------------------------
# Step 2: Feature Cleaning + Mapping Functions
# ---------------------------

REGION_MAP = {
    "Lebanon": ["Lebanon"],
    "Egypt": ["Egypt"],
    "UAE": ["UAE"],
    "Saudi Arabia": ["Saudi Arabia"],
    "Gulf countries (UAE, Saudi, Qatar, etc.)": ["Gulf", "UAE", "Saudi Arabia", "Qatar", "Kuwait", "Bahrain"],
    "Levant (Lebanon, Syria, Jordan, Palestine)": ["Levant", "Lebanon", "Syria", "Jordan", "Palestine"],
    "North Africa (Egypt, Tunisia, Morocco, Algeria, Libya)": ["North Africa", "Egypt", "Tunisia", "Morocco", "Algeria", "Libya"],
    "The MENA region overall": ["MENA", "Levant", "North Africa", "Gulf", "Lebanon", "Egypt", "UAE", "Saudi Arabia"],
    "Global-level news (like international events or global topics)": ["Global", "International"],
    "News from anywhere in the world": ["__match_all__"]
}

TONE_MAP = {
    "Straightforward and informative": "factual",
    "Emotional or personal stories": "emotional",
    "Positive and motivating": "inspirational",
    "Strong or direct in their message": "confrontational",
    "Relaxed and casual": "casual",
    "Funny or sarcastic": "satirical"
}

STYLE_MAP = {
    "A person telling a story": "storytelling",
    "Two or more people talking in an interview": "interview",
    "A reporter giving a news update": "report",
    "Someone casually explaining or reacting to something": "casual report",
    "Deeper videos like mini-documentaries": "documentary",
    "A video from a brand or campaign": "advertisement"
}

EMOTION_MAP = {
    "Gives me hope": "hope",
    "Makes me feel proud (culture, success, identity)": "pride",
    "Makes me happy or smile": "joy",
    "Something emotional or touching": "sadness",
    "Shows a serious problem or injustice": "anger",
    "Feels intense or heavy": "fear"
}

TOPIC_MAP = {
    "Technology & digital trends": "technology",
    "Business & economy": "economy",
    "Environment or climate": "environment",
    "Mental health & wellbeing": "health",
    "Social media & internet culture": "internet",
    "Politics & current issues": "politics",
    "Human rights or justice": "justice",
    "History or cultural stories": "history",
    "Art, music, or film": "art",
    "Fashion or lifestyle": "lifestyle",
    "Celebrities & entertainment": "entertainment",
    "Sports": "sports",
    "Education or learning": "education"
}

LANGUAGE_MAP = {
    "Arabic": "arabic",
    "English": "english",
    "I’m okay with both": "mixed"
}

FALLBACKS = [
    "I’m not sure",
    "I don’t really have a preference",
    "I just want the information",
    "I don’t mind how long it is",
    "I don’t really care how it’s presented",
    "News from anywhere in the world"
]

def clean_multiselect(raw_list, mapping_dict=None):
    cleaned = []
    for val in raw_list:
        if val in FALLBACKS:
            continue
        mapped = mapping_dict[val] if mapping_dict and val in mapping_dict else val
        cleaned.append(mapped)
    return cleaned

def clean_language(val):
    return LANGUAGE_MAP.get(val, None)

def count_selected_features(cleaned_dict):
    return sum(1 for val in cleaned_dict.values() if val and not (isinstance(val, str) and val.strip() == ""))

# ---------------------------
# Step 3: Title and Intro
# ---------------------------

st.title("Smart Video Recommender")
st.markdown("""
Welcome to your personalized news feed. Answer a few short questions and our system will recommend the best-fit videos based on your preferences.

*All videos come from a diverse, multilingual news platform tailored for the MENA region and beyond.*
""")

# ---------------------------
# Step 4: Form UI
# ---------------------------

with st.form("preference_form"):
    st.subheader("🎯 What would you like to watch?")

    tone_input = st.multiselect("Preferred tone:", list(TONE_MAP.keys()) + ["I don’t really have a preference"], key="tone_key")
    style_input = st.multiselect("Video style:", list(STYLE_MAP.keys()) + ["I don’t really care how it’s presented"], key="style_key")
    emotion_input = st.multiselect("Emotional vibe:", list(EMOTION_MAP.keys()) + ["I just want the information", "I’m not sure"], key="emotion_key")
    region_input = st.multiselect("Regions you're interested in:", list(REGION_MAP.keys()), key="region_key")
    duration_input = st.multiselect("Preferred video length:", [
        "Up to 1 minute", "1 to 2 minutes", "2 to 5 minutes", "More than 5 minutes", "I don’t mind how long it is"], key="duration_key")
    language_input = st.radio("Preferred language:", list(LANGUAGE_MAP.keys()), key="language_key")
    topic_input = st.multiselect("Topics that interest you:", list(TOPIC_MAP.keys()) + ["I’m not sure"], key="topic_key")

    submitted = st.form_submit_button("🔍 Show my personalized recommendations")

# ---------------------------
# Step 5: Clean + Encode + Predict + Match
# ---------------------------

if submitted:
    user_clean = {
        'tone': clean_multiselect(tone_input, TONE_MAP),
        'style': clean_multiselect(style_input, STYLE_MAP),
        'emotion': clean_multiselect(emotion_input, EMOTION_MAP),
        'region_tags': clean_multiselect(region_input),
        'duration_group': clean_multiselect(duration_input),
        'language': clean_language(language_input),
        'category': clean_multiselect(topic_input, TOPIC_MAP)
    }

    user_original = {
        'tone': tone_input,
        'style': style_input,
        'emotion': emotion_input,
        'region_tags': region_input,
        'duration_group': duration_input,
        'language': language_input,
        'category': topic_input
    }

    encoded_user = pd.DataFrame()
    multilabel_features = ['tone', 'style', 'emotion', 'region_tags', 'duration_group', 'category']

    for feature in multilabel_features:
        mlb = MultiLabelBinarizer()
        encoded = mlb.fit_transform([user_clean[feature]])
        temp_df = pd.DataFrame(encoded, columns=[f'u_{feature}_{cls}' for cls in mlb.classes_])
        encoded_user = pd.concat([encoded_user, temp_df], axis=1)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    language_df = pd.DataFrame(ohe.fit_transform([[user_clean['language']]]),
                               columns=[f'u_language_{cls}' for cls in ohe.categories_[0]])
    encoded_user = pd.concat([encoded_user, language_df], axis=1)

    encoded_user_final = encoded_user.reindex(columns=feature_columns, fill_value=0)
    _ = model.predict_proba(encoded_user_final)[0, 1]  # do not display

    matched_videos = []
    def is_real_preference(raw_input, mapping_dict=None):
        return bool(clean_multiselect(raw_input, mapping_dict))

    selected_feature_count = sum([
        is_real_preference(tone_input, TONE_MAP),
        is_real_preference(style_input, STYLE_MAP),
        is_real_preference(emotion_input, EMOTION_MAP),
        is_real_preference(region_input),
        is_real_preference(duration_input),
        language_input != "I’m okay with both",
        is_real_preference(topic_input, TOPIC_MAP)
    ])

    for i, row in videos_df.iterrows():
            row_lang = row['language'].lower().strip()
            user_lang = user_clean.get('language', None)
            if isinstance(user_lang, str):
                user_lang = user_lang.lower().strip()
            else:
                user_lang = None

            video_regions = eval(row['region_tags']) if isinstance(row['region_tags'], str) else []
            match_count = 0
            match_reasons = []
            matched_values = {}

            if selected_feature_count == 1 and user_lang and user_lang != 'mixed':
                if row_lang != user_lang:
                    continue

            if user_clean['tone']:
                matched_tones = [
                    original for original, cleaned in zip(user_original['tone'], user_clean['tone'])
                    if cleaned in row['tone']
                ]
                if matched_tones:
                    match_count += 1
                    match_reasons.append("Tone")
                    matched_values['Tone'] = ", ".join(matched_tones)

            if user_clean['style']:
                matched_styles = [
                    original for original, cleaned in zip(user_original['style'], user_clean['style'])
                    if cleaned in row['style']
                ]
                if matched_styles:
                    match_count += 1
                    match_reasons.append("Style")
                    matched_values['Style'] = ", ".join(matched_styles)

            if user_clean['emotion']:
                matched_emotions = [
                    original for original, cleaned in zip(user_original['emotion'], user_clean['emotion'])
                    if cleaned in row['emotion']
                ]
                if matched_emotions:
                    match_count += 1
                    match_reasons.append("Emotion")
                    matched_values['Emotion'] = ", ".join(matched_emotions)

            if user_clean['category']:
                matched_topics = [
                    original for original, cleaned in zip(user_original['category'], user_clean['category'])
                    if cleaned in row['category']
                ]
                if matched_topics:
                    match_count += 1
                    match_reasons.append("Topic")
                    matched_values['Topic'] = ", ".join(matched_topics)

            if user_clean['duration_group']:
                if 'I don’t mind how long it is' in duration_input:
                    match_count += 1
                    match_reasons.append("Any duration")
                    matched_values['Duration'] = "Any"
                elif row['duration_group'] in user_clean['duration_group']:
                    match_count += 1
                    match_reasons.append("Duration")
                    matched_values['Duration'] = row['duration_group']

            if user_lang and language_input != "I’m okay with both":
                if user_lang == 'mixed':
                    if row_lang in ['arabic', 'english', 'mixed']:
                        match_count += 1
                        match_reasons.append("Language")
                        matched_values['Language'] = row['language'].capitalize()
                elif user_lang in ['arabic', 'english']:
                    if row_lang == user_lang:
                        match_count += 1
                        match_reasons.append("Language")
                        matched_values['Language'] = row['language'].capitalize()


            if user_clean['region_tags']:
                if 'News from anywhere in the world' in region_input:
                    matched_values['Region'] = "Any"
                else:
                    matched_regions = []
                    for user_tag in user_original['region_tags']:
                        expanded_tags = set(REGION_MAP.get(user_tag, [user_tag]))
                        if any(region in expanded_tags for region in video_regions):
                            matched_regions.append(user_tag)
                    if matched_regions:
                        match_count += 1
                        match_reasons.append("Region")
                        matched_values['Region'] = ", ".join(matched_regions)

            if match_count > 0:
                matched_keys = [key.lower() for key in matched_values.keys()]
            
                unmatched_fields = [
                    display_name
                    for feature_key, display_name in [
                        ('tone', 'Tone'),
                        ('style', 'Style'),
                        ('emotion', 'Emotion'),
                        ('region_tags', 'Region'),
                        ('duration_group', 'Duration'),
                        ('language', 'Language'),
                        ('category', 'Topic')
                    ]
                    if user_clean.get(feature_key) and display_name.lower() not in matched_keys
                ]

                matched_videos.append({
                    'video_id': row['video_id'],
                    'title': row['title'],
                    'url': row['url'],
                    'score': match_count,
                    'reasons': match_reasons,
                    'total_selected': selected_feature_count,
                    'matched_values': matched_values,
                    'unmatched_fields': unmatched_fields
                })

    shuffle(matched_videos)
    matched_videos = sorted(matched_videos, key=lambda x: x['score'], reverse=True)
    st.session_state.last_results = matched_videos[:5]

# ---------------------------
# Step 6: Show Results (if present)
# ---------------------------

if st.session_state.last_results:
    st.markdown("---")
    st.subheader("📺 Top Recommended Videos for You")
    for vid in st.session_state.last_results:
        score = vid['score']
        total = vid.get('total_selected', 7)  # fallback just in case
        percent = int((score / total) * 100) if total > 0 else 0

        # color logic
        if percent < 50:
            color = "red"
        elif percent <= 70:
            color = "orange"
        else:
            color = "green"

        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-bottom: 10px;">
            <b>{vid['title']}</b><br>
            <i>Matched on:</i> {', '.join(vid['reasons'])}<br>
            <b>Score:</b> {score}/{total} &nbsp;&nbsp;
            <b style="color:{color}">({percent}% match)</b>
        </div>
        """, unsafe_allow_html=True)
        
        unmatched = vid.get('unmatched_fields', ['Tone', 'Topic'])


        matched_values = vid.get('matched_values', {})

        if matched_values:
            st.markdown("**🧩 Matched Features:**")
            for feature, value in matched_values.items():
                if value:
                    st.markdown(f"- **{feature}**: {value}")

        if unmatched:
            st.markdown(f"⚠️ Some preferences were not matched: {', '.join(unmatched)}.")
            st.markdown("_These preferences were not matched because no available videos currently meet all your selected criteria. You can broaden or adjust your answers to discover more relevant results._")

        st.video(vid['url'])

    st.markdown("---")
    if st.button("🔁 Reset Recommendations"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    with st.expander("⭐ Rate Your Recommendations"):
        rating = st.slider("How relevant were these results?", 1, 5, 3)
        if st.button("Submit Rating"):
            st.success(f"Thank you! You rated this {rating} out of 5.")

elif submitted:
    if selected_feature_count == 1:
        st.warning("We couldn’t find any strong matches based on a single preference. Try adding more preferences to get better recommendations.")
    else:
        st.warning("No videos matched your combination of preferences. Try broadening your selections.")