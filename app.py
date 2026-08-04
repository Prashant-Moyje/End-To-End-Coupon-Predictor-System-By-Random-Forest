import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (preprocessing + model bundled together)
model = joblib.load('coupon_model.pkl')

st.set_page_config(page_title="Coupon Acceptance Predictor", page_icon="🎟️")
st.title("🎟️ Coupon Acceptance Predictor")
st.write("Fill in the driving context below to predict whether the coupon will be accepted.")

# --- Input fields (based on the standard in-vehicle coupon dataset) ---
# NOTE: Update the options below to EXACTLY match the unique values in your DS_DATA.csv

destination = st.selectbox("Destination", ["Home", "Work", "No Urgent Place"])
passanger = st.selectbox("Passenger", ["Alone", "Friend(s)", "Kid(s)", "Partner"])
weather = st.selectbox("Weather", ["Sunny", "Rainy", "Snowy"])
temperature = st.selectbox("Temperature", [30, 55, 80])
coupon = st.selectbox("Coupon type", ["Restaurant(<20)", "Coffee House", "Carry out & Take away",
                                       "Bar", "Restaurant(20-50)"])
expiration = st.selectbox("Coupon expiration", ["2h", "1d"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.selectbox("Age", ["below21", "21", "26", "31", "36", "41", "46", "50plus"])
maritalStatus = st.selectbox("Marital Status", ["Single", "Married partner", "Unmarried partner",
                                                 "Divorced", "Widowed"])
has_children = st.selectbox("Has Children", [0, 1])
education = st.selectbox("Education", ["Some college - no degree", "Bachelors degree",
                                        "Graduate degree (Masters or Doctorate)",
                                        "Associates degree", "High School Graduate",
                                        "Some High School"])
occupation = st.selectbox("Occupation", [
    "Unemployed", "Architecture & Engineering", "Student",
    "Education&Training&Library", "Healthcare Support",
    "Healthcare Practitioners & Technical", "Sales & Related", "Management",
    "Arts Design Entertainment Sports & Media", "Computer & Mathematical",
    "Life Physical Social Science", "Personal Care & Service",
    "Community & Social Services", "Office & Administrative Support",
    "Construction & Extraction", "Legal", "Retired",
    "Installation Maintenance & Repair", "Transportation & Material Moving",
    "Business & Financial", "Protective Service",
    "Food Preparation & Serving Related", "Production Occupations",
    "Building & Grounds Cleaning & Maintenance", "Farming Fishing & Forestry"
])
income = st.selectbox("Income", ["Less than $12500", "$12500 - $24999", "$25000 - $37499",
                                  "$37500 - $49999", "$50000 - $62499", "$62500 - $74999",
                                  "$75000 - $87499", "$87500 - $99999", "$100000 or More"])

st.subheader("Past behavior frequency")
Bar = st.selectbox("Bar visits/month", ["never", "less1", "1~3", "4~8", "gt8"])
CoffeeHouse = st.selectbox("Coffee House visits/month", ["never", "less1", "1~3", "4~8", "gt8"])
CarryAway = st.selectbox("Carry-away visits/month", ["never", "less1", "1~3", "4~8", "gt8"])
RestaurantLessThan20 = st.selectbox("Restaurant(<20) visits/month", ["never", "less1", "1~3", "4~8", "gt8"])
Restaurant20To50 = st.selectbox("Restaurant(20-50) visits/month", ["never", "less1", "1~3", "4~8", "gt8"])

st.subheader("Trip direction")
toCoupon_GEQ5min = st.selectbox("Distance to coupon >= 5 min", [0, 1])
toCoupon_GEQ15min = st.selectbox("Distance to coupon >= 15 min", [0, 1])
toCoupon_GEQ25min = st.selectbox("Distance to coupon >= 25 min", [0, 1])
direction_same = st.selectbox("Coupon is in same direction as travel", [0, 1])
direction_opp = st.selectbox("Coupon is in opposite direction of travel", [0, 1])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "destination": destination,
        "passanger": passanger,
        "weather": weather,
        "temperature": temperature,
        "coupon": coupon,
        "expiration": expiration,
        "gender": gender,
        "age": age,
        "maritalStatus": maritalStatus,
        "has_children": has_children,
        "education": education,
        "occupation": occupation,
        "income": income,
        "Bar": Bar,
        "CoffeeHouse": CoffeeHouse,
        "CarryAway": CarryAway,
        "RestaurantLessThan20": RestaurantLessThan20,
        "Restaurant20To50": Restaurant20To50,
        "toCoupon_GEQ5min": toCoupon_GEQ5min,
        "toCoupon_GEQ15min": toCoupon_GEQ15min,
        "toCoupon_GEQ25min": toCoupon_GEQ25min,
        "direction_same": direction_same,
        "direction_opp": direction_opp,
    }])

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        if prediction == 1:
            st.success(f"✅ Likely to ACCEPT the coupon (confidence: {max(proba)*100:.1f}%)")
        else:
            st.error(f"❌ Likely to REJECT the coupon (confidence: {max(proba)*100:.1f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("This usually means the input column names/values don't match what the model was trained on. "
                "Check X.columns and X[col].unique() in your notebook and update this form accordingly.")
