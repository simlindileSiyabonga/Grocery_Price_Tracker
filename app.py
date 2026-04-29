import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# --- Load model and encoders ---
model_path = r"C:\Users\082nx\Documents\GitHub\Grocery_Price_Tracker\rf_model_combined.pkl"
with open(model_path, "rb") as f:
    data = pickle.load(f)

model = data['model']
le_store = data['le_store']
le_product = data['le_product']

# --- Load dataset ---
csv_path = r"C:\Users\082nx\Documents\GitHub\Grocery_Price_Tracker\combined_prices.csv"
df = pd.read_csv(csv_path)

# --- Streamlit Chatbot UI ---
st.title("🛒 Grocery Price Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about a product (e.g apples?') Cheapest Store will be displayed"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Detect product from dataset ---
    product = None
    for p in df['Product_Clean'].unique():
        if p.lower() in user_input.lower():
            product = p
            break

    if product:
        # Predict price for all stores using today's date
        today = datetime.now()
        year, month, dayofweek = today.year, today.month, today.weekday()

        predictions = []
        for store in df['Store'].unique():
            # Encode features
            store_encoded = le_store.transform([store])[0]
            product_encoded = le_product.transform([product])[0]

            features_df = pd.DataFrame([{
                'Store': store_encoded,
                'Product_Clean': product_encoded,
                'Year': year,
                'Month': month,
                'DayOfWeek': dayofweek  # Must match training feature
            }])

            price = model.predict(features_df)[0]
            predictions.append((store, price))

        results = pd.DataFrame(predictions, columns=["Store", "Predicted_Price"])
        lowest = results.loc[results['Predicted_Price'].idxmin()]

        bot_reply = (
            f"The store predicted to have the **lowest price** for **{product}** is "
            f"**{lowest['Store']}**, with an estimated price of **R{lowest['Predicted_Price']:.2f}**."
        )
    else:
        bot_reply = "I couldn’t identify the product. Please ask again, e.g., 'How much is bread at Checkers?'"

    # Display bot reply
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
