from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import numpy as np
import plotly.express as px
import pandas as pd

# تحميل الموديل والـ scaler
with open("rf_model.pkl", "rb") as f1:
    rf = pickle.load(f1)

with open("scaler.pkl", "rb") as f2:
    scaler = pickle.load(f2)

# تحميل الـ LabelEncoder (لو هتستخدمه في المستقبل)
with open("label_encoder.pkl", "rb") as f3:
    le = pickle.load(f3) 


# FastAPI app
app = FastAPI() 

# صفحة الإدخال (form)
@app.get("/", response_class=HTMLResponse)
def form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Air Quality Prediction</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #ffecd2, #fcb69f);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 130vh;
                margin: 0;
            }
          
            
            .card {
                margin-top:3px;
                width: 400px; /* عرض مناسب للفورم */
                background: #fff;
                padding: 30px 40px;
                border-radius: 16px;
                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                text-align: center;
                animation: fadeIn 1s ease-in-out;
            }


            @keyframes fadeIn {
                from {opacity: 0; transform: translateY(-20px);}
                to {opacity: 1; transform: translateY(0);}
            }

            h2 {
                margin-top: 3px;
                margin-bottom: 8px;
                font-size: 24px;
                color: #333;
            }

            /* ده خلي الـ form في النص وعرضه مناسب */
            form {  
                max-width: 600px;   /* العرض الأقصى للفورم */
                margin: 30px auto;  /* يخليه في النص */
                padding: 20px;
                background: #fff;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                
            }

/* ده خلي الـ labels تبقى فوق الـ input */
            label {
               display: block;
               margin-bottom: 6px;
               font-weight: bold;
            }

/* ده يخلي الـ input أكبر ومقاسه واحد */
            input[type="text"], input[type="number"] {
                width: 100%;       /* كل input ياخد عرض الفورم */
                padding: 10px;
                margin-bottom: 15px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }

/* زرار predict */
            button {
                width: 100%;
                padding: 12px;
                background: #ff7b54;
                color: white;
                font-size: 18px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
            }

            button:hover {
                background: #ff5722;
            }

            
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🌍 Air Quality Prediction</h2>
            <form action="/predict" method="post">
                <label for="Temperature">Temperature:</label>
                <input type="number" step="any" name="Temperature" required>

                <label for="Humidity">Humidity:</label>
                <input type="number" step="any" name="Humidity" required>

                <label for="PM2_5">PM2_5:</label>
                <input type="number" step="any" name="PM2_5" required>

                <label for="PM10">PM10:</label>
                <input type="number" step="any" name="PM10" required>

                <label for="NO2">NO2:</label>
                <input type="number" step="any" name="NO2" required>

                <label for="SO2">SO2:</label>
                <input type="number" step="any" name="SO2" required>
                
                <label for="CO">CO:</label>
                <input type="number" step="any" name="CO" required>

                <label for="Proximity">Proximity to Industrial Areas:</label>
                <input type="number" step="any" name="Proximity" required>

                <label for="Population">Population Density:</label>
                <input type="number" step="any" name="Population" required>

                <button type="submit">Predict</button>
            </form>

        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# صفحة النتيجة (prediction result)
@app.post("/predict", response_class=HTMLResponse)
def predict( 
    Temperature: float = Form(...),
    Humidity: float = Form(...),
    PM2_5: float = Form(...),
    PM10: float = Form(...),
    NO2: float = Form(...),
    SO2: float = Form(...),
    CO: float = Form(...),
    Proximity: float = Form(...),
    Population : float = Form(...)
):
    
    features = np.array([[Temperature, Humidity, PM2_5, PM10, NO2, SO2, CO, Proximity, Population]])
    features_scaled = scaler.transform(features)

    
    prediction = rf.predict(features_scaled)[0]

    result = le.inverse_transform([prediction])[0]

    # HTML للنتيجة
    result_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #74ebd5, #9face6);
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .result-card {{
                background: #fff;
                padding: 30px 40px;
                border-radius: 15px;
                box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
                text-align: center;
                animation: fadeIn 1s ease-in-out;
            }}
            @keyframes fadeIn {{
                from {{opacity: 0; transform: scale(0.9);}}
                to {{opacity: 1; transform: scale(1);}}
            }}
            h2 {{
                color: #333;
            }}
            .btn {{
                margin-top: 20px;
                display: inline-block;
                padding: 10px 20px;
                background: #4CAF50;
                color: #fff;
                text-decoration: none;
                border-radius: 8px;
                transition: 0.3s;
            }}
            .btn:hover {{
                background: #45a049;
            }}
        </style>
    </head>
    <body>
        <div class="result-card">
            <h2>Predicted Air Quality: {result}</h2>
            <a href="/" class="btn">🔙 Back</a>
        </div>
    </body>
    </html>
    """


    return HTMLResponse(result_html)





# تشغيل على Colab
# nest_asyncio.apply()
# public_url = ngrok.connect(8000)
# print("Public URL:", public_url)
# uvicorn.run(app, host="0.0.0.0", port=8000)


#   .card {
#                 background: #fff;
#                 padding: 40px 50px;
#                 border-radius: 15px;
#                 box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
#                 text-align: center;
#                 width: 400px;
#                 animation: fadeIn 1s ease-in-out;
#   }