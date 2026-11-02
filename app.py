
import pickle
import streamlit as st

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="centered"
)

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}
.block-container {
    max-width: 760px;
    padding-top: 2.5rem;
}
.hero {
    background: linear-gradient(120deg, #312e81, #4f46e5);
    padding: 30px;
    border-radius: 22px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #e0e7ff;
    font-size: 16px;
}
.stTextArea textarea {
    border-radius: 12px;
    background: white;
}
.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-weight: 600;
}
.result {
    padding: 20px;
    border-radius: 14px;
    margin-top: 15px;
}
.spam {
    background: #fee2e2;
    border: 1px solid #f87171;
    color: #991b1b;
}
.genuine {
    background: #dcfce7;
    border: 1px solid #4ade80;
    color: #166534;
}
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛡️ SpamGuard AI</h1>
    <p>Smart message detection powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Analyze Your Message")
st.write("Paste a message below to check whether it looks like spam.")

message = st.text_area(
    "Message",
    height=150,
    max_chars=2000,
    placeholder="Example: Congratulations! Claim your free prize now..."
)

st.caption(f"Characters: {len(message)} / 2000")

col1, col2 = st.columns(2)

with col1:
    check = st.button("🔍 Analyze Message", type="primary")

with col2:
    clear = st.button("🗑️ Clear Message")

if clear:
    st.rerun()

if check:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities) * 100

        if prediction == "spam":
            st.markdown(f"""
            <div class="result spam">
                <h3>🚨 This Message Looks Like Spam</h3>
                <p>Be careful. Avoid clicking suspicious links or sharing personal details.</p>
                <b>Model confidence: {confidence:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result genuine">
                <h3>✅ This Message Looks Genuine</h3>
                <p>The model classified this message as a normal message.</p>
                <b>Model confidence: {confidence:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "This is a machine-learning prediction, not a guarantee. "
            "Always be cautious with unknown senders and links."
        )

st.markdown("""
<div class="footer">
    Built with Python • Scikit-learn • Streamlit<br>
    CodeAlpha Internship Project
</div>
""", unsafe_allow_html=True)
