#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Model Imported

model = joblib.load("car_price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_csv(r"used_car_price_prediction_dataset.csv")

# Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)
st.markdown("""
<style>

/* Reduce top padding */
.block-container{
    padding-top: 3.5rem;
    padding-bottom: 1rem;
}

/* Remove extra space above the page */
[data-testid="stAppViewContainer"]{
    margin-top: 0rem;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.navbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:#1f2937;
    padding:16px 30px;
    border-radius:14px;
    margin-bottom:25px;
    box-shadow:0 4px 12px rgba(0,0,0,.2);
}

.logo{
    color:white;
    font-size:34px;
    font-weight:700;
}

.menu{
    display:flex;
    gap:35px;
}

.menu a{
    color:white;
    text-decoration:none;
    font-size:18px;
    font-weight:500;
    transition:.3s;
}

.menu a:hover{
    color:#22c55e;
}

@media (max-width: 992px){

    .navbar{
        flex-direction:column;
        align-items:center;
        gap:18px;
    }

    .logo{
        font-size:28px;
        text-align:center;
        white-space:normal;
    }

    .menu{
        justify-content:center;
        flex-wrap:wrap;
        gap:18px;
    }

    .menu a{
        font-size:16px;
    }

}

@media (max-width:576px){

    .logo{
        font-size:22px;
        text-align:center;
    }

    .menu{
        display:none;
    }

}

/* ============================= */
/* Prediction Card Responsive */
/* ============================= */

.prediction-card{
    background:linear-gradient(135deg,#14532D,#166534);
    padding:35px;
    border-radius:18px;
    color:white;
    margin-top:20px;
    box-shadow:0 6px 20px rgba(0,0,0,.25);
}

.prediction-title{
    font-size:38px;
    font-weight:700;
    margin-bottom:15px;
}

.prediction-text{
    font-size:19px;
    color:#E5E7EB;
}

.price-label{
    font-size:22px;
    font-weight:600;
}

.price-value{
    font-size:60px;
    font-weight:700;
    color:white;
    margin:10px 0;
}

.price-full{
    font-size:24px;
    color:#D1FAE5;
}

.model-info{
    font-size:18px;
}

.disclaimer{
    font-size:17px;
    color:#D1FAE5;
    font-style:italic;
}

/* ============================= */
/* Tablet */
/* ============================= */

@media (max-width:992px){

.prediction-card{
    padding:28px;
}

.prediction-title{
    font-size:30px;
}

.price-value{
    font-size:46px;
}

.price-full{
    font-size:20px;
}

.model-info,
.disclaimer,
.prediction-text{
    font-size:16px;
}

}

/* ============================= */
/* Mobile */
/* ============================= */

@media (max-width:576px){

.prediction-card{
    padding:20px;
    border-radius:14px;
}

.prediction-title{
    font-size:20px;
}

.prediction-text{
    font-size:14px;
}

.price-label{
    font-size:16px;
}

.price-value{
    font-size:34px;
    line-height:1.2;
}

.price-full{
    font-size:17px;
}

.model-info{
    font-size:15px;
}

.disclaimer{
    font-size:14px;
}

}
</style>

<div class="navbar">

<div class="logo">
🚗 Car Price Predictor
</div>

<div class="menu">
<a href="#">🏠 Home</a>
<a href="#predict">🔮 Predict</a>
<a href="#signin">👤 Sign Up</a>
</div>

</div>
""", unsafe_allow_html=True)
 
# st.markdown("""
# <div style="
# background: linear-gradient(135deg, #14532D, #16A34A);
# padding:30px;
# border-radius:18px;
# text-align:center;
# margin-bottom:20px;
# ">
# <h1 style="color:white;margin:0;">
# 🚗 Used Car Price Prediction
# </h1>
# <p style="color:white;font-size:22px;margin-top:10px;">
# Predict the car value of a used car using Machine Learning
# </p>
# </div>
# """, unsafe_allow_html=True)

#title

st.success(
    "This application estimates the price of new/used cars using a machine learning model trained on 39,298 real vehicle records."
)

# ===========================
# Sidebar
# ===========================

st.sidebar.title("🚗 Car Price Predictor")

st.sidebar.success("✅ Model Ready")

st.sidebar.write("### 🤖 Model")
st.sidebar.write("Tuned XGBoost")

st.sidebar.write("### 📈 Accuracy")
st.sidebar.metric("R² Score", "0.935")

st.sidebar.write("### 📂 Dataset")
st.sidebar.write(f"{len(df):,} Cars")

st.sidebar.write("### 🔢 Features")
st.sidebar.write("9 Input Features")
st.sidebar.divider()
st.sidebar.markdown("""
### 📌 About

This application predicts the price of a new/used car using a **Tuned XGBoost Regression Model**.

### 🧠 Machine Learning Model
- XGBoost Regressor
- Hyperparameter Tuned
- R² Score: **0.935**

### 📊 Features Used
- Brand
- Model
- Fuel Type
- Transmission
- Owner
- Age
- Engine
- Mileage
- Kilometers Driven

---
Developed By - Nishant (BCA-III) 
""")

# st.title("🚗 Used Car Price Prediction")
# st.write("Predict the resale value of a used car using Machine Learning.")

st.write("")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(df["Brand"].unique()))

    models = sorted(
    df[df["Brand"] == brand]["model"].unique()
    )

    model_name = st.selectbox("Model", models)
    
    transmission = st.selectbox(
    "Transmission",
    sorted(df["Transmission"].unique())
    )
    
    owner = st.selectbox(
        "Owner",
        ['First','Second']
    )
    
    age = st.number_input(
        "Age (Years)",
        min_value=0,
        max_value=30,
        value=5
    )

with col2:
    fuel = st.selectbox("Fuel Type", sorted(df["FuelType"].unique()))
    km = st.number_input("Kilometers Driven", 0, 500000, 50000)
    engine = st.number_input("Engine (CC)", 500, 5000, 1200)
    mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, 20.0)
    
st.divider()

if st.button("🚗 Predict Price",use_container_width=True,type="primary"):

    input_data = {
        "Brand": brand,
        "model": model_name,
        "Age": age,
        "kmDriven": km,
        "Transmission": transmission,
        "Owner": owner.lower(),
        "FuelType": fuel,
        "engine": engine,
        "mileage": mileage
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = round(float(model.predict(input_df)[0]))
    price_lakh = prediction / 100000
    
    from datetime import datetime

    st.session_state.history.insert(0,{
    "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),
    "Brand": brand,
    "Model": model_name,
    "Fuel": fuel,
    "Owner": owner,
    "Price": f"₹{price_lakh:.2f} Lakhs"
    })

# Keep only last 10 predictions
    st.session_state.history = st.session_state.history[:10]
    
    st.subheader("Selected Car Details")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"🚗 **Brand:** {brand}")
        st.markdown(f"📌 **Model:** {model_name}")
        st.markdown(f"⛽ **Fuel Type:** {fuel}")
        st.markdown(f"⚙️ **Transmission:** {transmission}")
        st.markdown(f"👤 **Owner:** {owner}")
    with c2:
        st.markdown(f"📅 **Age:** {age} Years")
        st.markdown(f"🔧 **Engine:** {engine:,} CC")
        st.markdown(f"🏁 **Mileage:** {mileage} km/l")
        st.markdown(f"🛣️ **Kilometers Driven:** {km:,}")
    

    st.markdown(f"""
    <div class="prediction">

    <h2 class="prediction-title">
    📋 Prediction Summary
    </h2>

    <p class="prediction-text">
    Based on the selected specifications, the estimated value of this vehicle is:
    </p>

    <p class="price-label">
    💰 Estimated Car Value
    </p>

    <h1 class="price-value">
    ₹{price_lakh:.2f} Lakhs
    </h1>

    <p class="price-full">
    (₹{prediction:,.0f})
    </p>

    <hr style="
    border:none;
    height:1px;
    background:rgba(255,255,255,0.25);
    margin:25px 0;
    ">

    <p class="model-info">
    ✅ <b>Model Used:</b> Tuned XGBoost Regressor
    </p>

    <p class="model-info">
    📈 <b>Model Accuracy (R²):</b> <b>93.5%</b>
    </p>

    <p class="disclaimer">
    This prediction is based on historical car market data and should be considered an estimate.
    </p>

    </div>
    """, unsafe_allow_html=True)
    # st.info(f"""
    # ## 📋 Prediction Summary

    # Based on the selected specifications, the estimated resale value of this vehicle is:

    # # 💰 ₹{prediction:,.0f}  ≈  ₹{price_lakh:.2f} Lakhs

    # ---

    # ✅ **Model Used:** Tuned XGBoost Regressor

    # 📈 **Model Accuracy (R²):** **93.5%**

    # *This prediction is based on historical car market data and should be considered an estimate.*
    # """)
    from io import BytesIO
    from datetime import datetime

    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    title = styles["Title"]
    title.alignment = TA_CENTER

    heading = styles["Heading2"]
    normal = styles["BodyText"]

    story = []

    # ==========================
    # Title
    # ==========================

    story.append(Paragraph("Car Price Prediction Report", title))
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d %B %Y  %I:%M %p')}",
            normal,
        )
    )

    story.append(Spacer(1,20))

    # ==========================
    # Vehicle Details Table
    # ==========================

    table_data = [

    ["Brand",brand],
    ["Model",model_name],
    ["Fuel Type",fuel],
    ["Transmission",transmission],
    ["Owner",owner],
    ["Age",f"{age} Years"],
    ["Engine",f"{engine:,} CC"],
    ["Mileage",f"{mileage} km/l"],
    ["Kilometers Driven",f"{km:,} km"]

    ]

    table = Table(table_data,colWidths=[170,250])

    table.setStyle(TableStyle([

    ("BACKGROUND",(0,0),(0,-1),colors.darkgreen),
    ("TEXTCOLOR",(0,0),(0,-1),colors.white),

    ("BACKGROUND",(1,0),(1,-1),colors.whitesmoke),

    ("GRID",(0,0),(-1,-1),1,colors.grey),

    ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

    ("BOTTOMPADDING",(0,0),(-1,-1),8),

    ]))

    story.append(table)

    story.append(Spacer(1,25))

    # ==========================
    # Prediction
    # ==========================

    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT

    price_style = ParagraphStyle(
        "Price",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,          # line spacing
        textColor=colors.darkgreen,
        alignment=TA_LEFT,
    )

    amount_style = ParagraphStyle(
        "Amount",
        parent=styles["Normal"],
        fontSize=16,
        leading=20,
        textColor=colors.grey,
    )

    story.append(Paragraph(f"INR - {price_lakh:.2f} Lakhs (Approx)", price_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"(INR - {prediction:,.0f})", amount_style))
    story.append(Spacer(1, 20))

    story.append(Spacer(1,20))

    # ==========================
    # Model Information
    # ==========================

    story.append(Paragraph("<b>Machine Learning Model</b>",heading))

    story.append(Paragraph("Model : Tuned XGBoost Regressor",normal))

    story.append(Paragraph("Prediction Accuracy (R² Score) : 93.5%",normal))

    story.append(Spacer(1,15))

    # ==========================
    # Disclaimer
    # ==========================

    story.append(Paragraph("<b>Disclaimer</b>",heading))

    story.append(
    Paragraph(
    """
    This predicted car value is generated using a Machine Learning model trained on
    39,298 historical car records. The estimate is intended for informational
    purposes only. Actual car prices may vary depending on vehicle condition,
    location, demand, service history, insurance claims, negotiations, and market trends.
    """,
    normal
    )
    )

    story.append(Spacer(1,30))

    # ==========================
    # Footer
    # ==========================

    footer = Paragraph(
    """
    <hr/>
    <center>
    <b>Car Price Prediction System</b><br/>
    Built using Python • Streamlit • XGBoost<br/>
    Developed by <b>Nishant (BCA-III)</b>
    </center>
    """,
    normal
    )

    story.append(footer)

    doc.build(story)

    buffer.seek(0)
    st.download_button(
        "📄 Download Prediction Report (PDF)",
        data=buffer,
        file_name="Car_Price_Prediction_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.divider()

st.subheader("🕒 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True
    )

    if st.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No predictions yet.")

st.divider()

st.markdown("""
<div style="text-align:center;color:gray;font-size:15px;margin-top:15px;">

© 2026 Car Price Prediction<br>

Built with ❤️ using Python • Streamlit • XGBoost<br>

Developed by <b>Nishant</b>

</div>
""", unsafe_allow_html=True)
