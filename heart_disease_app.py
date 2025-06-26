import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Risk Prediction")
st.markdown("---")

# Create top horizontal bar for inputs
st.markdown("### 📊 Health Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.slider("Age", min_value=20, max_value=100, value=50, help="Enter your age")
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="Select your biological sex"
    )
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina", 
            3: "Non-anginal Pain",
            4: "Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )

with col2:
    resting_bp = st.slider(
        "Resting Blood Pressure (mm Hg)",
        min_value=90, max_value=200, value=120,
        help="Systolic blood pressure at rest"
    )
    cholesterol = st.slider(
        "Cholesterol (mg/dl)",
        min_value=100, max_value=600, value=200,
        help="Serum cholesterol level"
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        options=[0, 1],
        format_func=lambda x: "≤ 120 mg/dl" if x == 0 else "> 120 mg/dl",
        help="Fasting blood sugar level"
    )

with col3:
    resting_ecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x],
        help="Electrocardiographic results at rest"
    )
    max_hr = st.slider(
        "Maximum Heart Rate",
        min_value=60, max_value=202, value=150,
        help="Maximum heart rate achieved during exercise"
    )
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Angina induced by exercise"
    )

with col4:
    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        min_value=-3.0, max_value=7.0, value=0.0, step=0.1,
        help="ST depression induced by exercise relative to rest"
    )
    st_slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat", 
            2: "Downsloping"
        }[x],
        help="Slope of peak exercise ST segment"
    )
    predict_button = st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True)

st.markdown("---")

@st.cache_resource
def load_model():
    """Load or create a simple model for demonstration"""
    model = RandomForestClassifier(n_estimators=50, max_depth=6, min_samples_split=8, min_samples_leaf=4, random_state=42)
    
    # Create synthetic data for training
    np.random.seed(42)
    n_samples = 1000
    
    age_data = np.random.normal(55, 10, n_samples)
    sex_data = np.random.choice([0, 1], n_samples)
    chest_pain_data = np.random.choice([1, 2, 3, 4], n_samples)
    resting_bp_data = np.random.normal(130, 20, n_samples)
    cholesterol_data = np.random.normal(250, 50, n_samples)
    fasting_bs_data = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    resting_ecg_data = np.random.choice([0, 1, 2], n_samples)
    max_hr_data = np.random.normal(150, 20, n_samples)
    exercise_angina_data = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    oldpeak_data = np.random.normal(1, 1.5, n_samples)
    st_slope_data = np.random.choice([0, 1, 2], n_samples)
    
    # Create target variable with more realistic logic
    target_data = (
        (age_data > 60).astype(int) * 0.4 +
        (resting_bp_data > 140).astype(int) * 0.3 +
        (cholesterol_data > 300).astype(int) * 0.3 +
        (exercise_angina_data) * 0.5 +
        (chest_pain_data == 4).astype(int) * 0.4 +
        (oldpeak_data > 2).astype(int) * 0.4 +
        (fasting_bs_data == 1).astype(int) * 0.2 +
        (resting_ecg_data > 0).astype(int) * 0.2 +
        (st_slope_data == 2).astype(int) * 0.3
    )
    
    # Add noise to make classification more challenging
    noise = np.random.normal(0, 0.25, n_samples)
    target_data = (target_data + noise) > 0.6
    
    X = np.column_stack([
        age_data, sex_data, chest_pain_data, resting_bp_data,
        cholesterol_data, fasting_bs_data, resting_ecg_data,
        max_hr_data, exercise_angina_data, oldpeak_data, st_slope_data
    ])
    
    model.fit(X, target_data)
    return model

@st.cache_resource
def train_all_models():
    """Train all models using the complete dataset"""
    try:
        # Try to load the expanded dataset first, fallback to original
        try:
            df = pd.read_csv('heart_disease_dataset.csv')
            
        except FileNotFoundError:
            df = pd.read_csv('heart_disease_dataset.csv')
            st.info("ℹ️ Using original training dataset")
        
        df = df.drop_duplicates()
        
        X = df.drop("target", axis=1)
        y = df["target"]
        
        # Use ALL data for training (no test split)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {
            "SVM": SVC(probability=True, C=8.0, kernel='rbf', gamma='scale'),
            "KNN": KNeighborsClassifier(n_neighbors=9, weights='uniform'),
            "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=12, min_samples_leaf=6, random_state=42)
        }
        
        results = []
        trained_models = {}
        
        for name, model in models.items():
            # Train on ALL data
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            # Calculate metrics on training data (since we're using all data)
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
            results.append({
                'Model': name,
                'Accuracy': round(acc, 3),
                'Precision': round(prec, 3),
                'Recall': round(rec, 3),
                'F1-Score': round(f1, 3)
            })
            
            trained_models[name] = model
        
        # For confusion matrices, we'll use the same data
        return results, trained_models, scaler, X_scaled, y
        
    except FileNotFoundError:
        st.error("Dataset not found. Using synthetic data for demonstration.")
        return None, None, None, None, None

model = load_model()
results, trained_models, scaler, X_scaled, y = train_all_models()

# Initialize results_df
results_df = None
if results is not None:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)

@st.cache_resource
def predict_heart_disease(input_data):
    """Make prediction using the model"""
    prediction = model.predict([input_data])
    probability = model.predict_proba([input_data])
    return prediction[0], probability[0]

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🎯 Prediction", "📊 Model Performance", "📈 Risk Analysis", "🔍 Model Comparison", "📋 Confusion Matrices", "🏆 Best Model", "🚨 High Risk Guide"])

# Initialize input_data
input_data = None

with tab1:
    if predict_button:
        input_data = [
            age, sex, chest_pain_type, resting_bp, cholesterol,
            fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
        ]
        
        prediction, probability = predict_heart_disease(input_data)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if prediction == 1:
                st.error("🚨 **HIGH RISK** - Heart Disease Detected")
                st.markdown(f"**Risk Probability:** {probability[1]:.1%}")
                st.markdown("""
                **Recommendations:**
                - Consult a cardiologist immediately
                - Monitor your blood pressure regularly
                - Adopt a heart-healthy diet
                - Exercise regularly under medical supervision
                - Avoid smoking and excessive alcohol
                """)
            else:
                st.success("✅ **LOW RISK** - No Heart Disease Detected")
                st.markdown(f"**Risk Probability:** {probability[0]:.1%}")
                st.markdown("""
                **Recommendations:**
                - Continue maintaining a healthy lifestyle
                - Regular check-ups with your doctor
                - Balanced diet and regular exercise
                - Monitor your health parameters
                """)
    else:
        st.info("👆 Please adjust the parameters above and click 'Predict Heart Disease Risk' to get your assessment.")

with tab2:
    st.subheader("📊 Model Performance Metrics")
    
    if results is not None and results_df is not None:
        st.dataframe(results_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            x = np.arange(len(metrics))
            width = 0.15
            
            for i, model_name in enumerate(results_df['Model'][:5]):
                values = [results_df.iloc[i][metric] for metric in metrics]
                ax.bar(x + i*width, values, width, label=model_name)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width*2)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            best_model = results_df.iloc[0]
            st.metric("Best Model", best_model['Model'])
            st.metric("Accuracy", f"{best_model['Accuracy']:.3f}")
            st.metric("Precision", f"{best_model['Precision']:.3f}")
            st.metric("Recall", f"{best_model['Recall']:.3f}")
            st.metric("F1-Score", f"{best_model['F1-Score']:.3f}")
    else:
        st.warning("Dataset not available. Performance metrics are based on synthetic data.")

with tab3:
    st.subheader("📈 Risk Factor Analysis")
    
    if predict_button:
        risk_factors = []
        
        if age > 60:
            risk_factors.append(("Age", "High", "Age over 60 increases risk", 0.8))
        elif age > 45:
            risk_factors.append(("Age", "Moderate", "Age 45-60 moderate risk", 0.5))
        else:
            risk_factors.append(("Age", "Low", "Age under 45 low risk", 0.2))
        
        if resting_bp > 140:
            risk_factors.append(("Blood Pressure", "High", "Systolic BP > 140 mm Hg", 0.9))
        elif resting_bp > 120:
            risk_factors.append(("Blood Pressure", "Moderate", "Systolic BP 120-140 mm Hg", 0.6))
        else:
            risk_factors.append(("Blood Pressure", "Low", "Normal blood pressure", 0.2))
        
        if cholesterol > 300:
            risk_factors.append(("Cholesterol", "High", "Cholesterol > 300 mg/dl", 0.8))
        elif cholesterol > 240:
            risk_factors.append(("Cholesterol", "Moderate", "Cholesterol 240-300 mg/dl", 0.5))
        else:
            risk_factors.append(("Cholesterol", "Low", "Normal cholesterol levels", 0.2))
        
        if exercise_angina == 1:
            risk_factors.append(("Exercise Angina", "High", "Exercise-induced chest pain", 0.9))
        else:
            risk_factors.append(("Exercise Angina", "Low", "No exercise-induced chest pain", 0.1))
        
        for factor, risk_level, description, risk_score in risk_factors:
            col1, col2 = st.columns([2, 3])
            with col1:
                if risk_level == "High":
                    st.markdown(f"🔴 **{factor}:** {risk_level}")
                elif risk_level == "Moderate":
                    st.markdown(f"🟡 **{factor}:** {risk_level}")
                else:
                    st.markdown(f"🟢 **{factor}:** {risk_level}")
            with col2:
                st.progress(risk_score)
                st.caption(description)
        
        overall_risk = sum([rf[3] for rf in risk_factors]) / len(risk_factors)
        st.markdown("---")
        st.subheader("Overall Risk Assessment")
        st.progress(overall_risk)
        st.metric("Overall Risk Score", f"{overall_risk:.1%}")
        
        if overall_risk > 0.7:
            st.error("High overall risk - Consult a healthcare provider")
        elif overall_risk > 0.4:
            st.warning("Moderate risk - Monitor your health closely")
        else:
            st.success("Low overall risk - Continue healthy lifestyle")
    else:
        st.info("👆 Please make a prediction first to see risk analysis.")

with tab4:
    st.subheader("🔍 Model Comparison")
    
    if results_df is not None and not results_df.empty and trained_models is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Rankings")
            for i, (_, row) in enumerate(results_df.iterrows()):
                if i == 0:
                    st.markdown(f"🥇 **{row['Model']}** - F1: {row['F1-Score']:.3f}")
                elif i == 1:
                    st.markdown(f"🥈 **{row['Model']}** - F1: {row['F1-Score']:.3f}")
                elif i == 2:
                    st.markdown(f"🥉 **{row['Model']}** - F1: {row['F1-Score']:.3f}")
                else:
                    st.markdown(f"**{row['Model']}** - F1: {row['F1-Score']:.3f}")
        
        with col2:
            if predict_button and scaler is not None:
                input_data_scaled = scaler.transform([input_data])
                best_model_name = results_df.iloc[0]['Model']
                best_model = trained_models[best_model_name]
                prediction_best = best_model.predict(input_data_scaled)
                
                # Handle probability prediction with error handling
                try:
                    probability_best = best_model.predict_proba(input_data_scaled)
                    if prediction_best[0] == 1:
                        st.error(f"Risk: {probability_best[0][1]:.1%}")
                    else:
                        st.success(f"Risk: {probability_best[0][0]:.1%}")
                except AttributeError:
                    # If predict_proba is not available, show prediction only
                    if prediction_best[0] == 1:
                        st.error("Risk: High (Probability not available)")
                    else:
                        st.success("Risk: Low (Probability not available)")
                
                st.markdown(f"### Best Model Prediction ({best_model_name})")
        
        if 'Random Forest' in trained_models:
            rf_model = trained_models['Random Forest']
            feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol', 
                           'Fasting BS', 'Resting ECG', 'Max HR', 'Exercise Angina', 'Oldpeak', 'ST Slope']
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.markdown("### Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='Importance', y='Feature')
            plt.title('Feature Importance in Heart Disease Prediction')
            st.pyplot(fig)
    else:
        st.warning("Dataset not available for detailed model comparison.")

with tab5:
    # st.subheader("📋 Confusion Matrices")
    
    if results is not None and trained_models is not None and X_scaled is not None and y is not None:
        # All models confusion matrices (sorted by performance, excluding best model)
        st.markdown("### 📊 Models Confusion Matrices")
        
        # Sort models by accuracy (excluding the best model)
        model_performance = []
        for model_name, model in trained_models.items():
            y_pred = model.predict(X_scaled)
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            model_performance.append((model_name, model, accuracy))
        
        # Sort by accuracy (descending) and remove the best model
        model_performance.sort(key=lambda x: x[2], reverse=True)
        # Remove the best model (first in the sorted list)
        remaining_models = model_performance[1:]
        
        # Create a grid layout for remaining confusion matrices
        for i in range(0, len(remaining_models), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name, model, accuracy = remaining_models[i]
                y_pred = model.predict(X_scaled)
                cm = confusion_matrix(y, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
                disp.plot(ax=ax, cmap='Blues', values_format='d')
                plt.title(f'{model_name}')
                st.pyplot(fig)
                
                # Display key metrics
                tn, fp, fn, tp = cm.ravel()
                st.metric(f"{model_name} Accuracy", f"{accuracy:.3f}")
            
            with col2:
                if i + 1 < len(remaining_models):
                    model_name, model, accuracy = remaining_models[i + 1]
                    y_pred = model.predict(X_scaled)
                    cm = confusion_matrix(y, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
                    disp.plot(ax=ax, cmap='Blues', values_format='d')
                    plt.title(f'{model_name}')
                    st.pyplot(fig)
                    
                    # Display key metrics
                    st.metric(f"{model_name} Accuracy", f"{accuracy:.3f}")
        
        # Summary table of all confusion matrix metrics (excluding best model)
        st.markdown("---")
        st.markdown("### 📈 Confusion Matrix Summary")
        
        summary_data = []
        for model_name, model, accuracy in remaining_models:
            y_pred = model.predict(X_scaled)
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            summary_data.append({
                'Model': model_name,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'TP': tp,
                'Sensitivity': round(sensitivity, 3),
                'Specificity': round(specificity, 3),
                'Precision': round(precision, 3),
                'Accuracy': round(accuracy, 3)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        st.dataframe(summary_df, use_container_width=True)
        
    else:
        st.warning("Dataset not available for confusion matrices.")
        st.info("Confusion matrices show the performance of each model in terms of True Positives, False Positives, True Negatives, and False Negatives.")

with tab6:
    st.subheader("🏆 Best Model")
    
    if results_df is not None and not results_df.empty and trained_models is not None and X_scaled is not None and y is not None:
        # Get the best model
        best_model_name = results_df.iloc[0]['Model']
        best_model = trained_models[best_model_name]
        y_pred_best = best_model.predict(X_scaled)
        cm_best = confusion_matrix(y, y_pred_best)
        
        st.markdown(f"### 🥇 **{best_model_name}** - Best Performing Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=['No Heart Disease', 'Heart Disease'])
            disp.plot(ax=ax, cmap='Blues', values_format='d')
            plt.title(f'Confusion Matrix - {best_model_name}')
            st.pyplot(fig)
        
        with col2:
            # Display metrics
            tn, fp, fn, tp = cm_best.ravel()
            
            # Create two columns for metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown("**Basic Metrics:**")
                st.metric("True Negatives (TN)", tn)
                st.metric("False Positives (FP)", fp)
                st.metric("False Negatives (FN)", fn)
                st.metric("True Positives (TP)", tp)
            
            with metric_col2:
                # Calculate additional metrics
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                
                st.markdown("**Performance Metrics:**")
                st.metric("Sensitivity (Recall)", f"{sensitivity:.3f}")
                st.metric("Specificity", f"{specificity:.3f}")
                st.metric("Precision", f"{precision:.3f}")
                st.metric("Accuracy", f"{accuracy:.3f}")
        
        # Show best model performance in comparison
        st.markdown("---")
        st.markdown("### 📊 Best Model Performance Summary")
        
        best_row = results_df.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{best_row['Accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{best_row['Precision']:.3f}")
        with col3:
            st.metric("Recall", f"{best_row['Recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{best_row['F1-Score']:.3f}")
        
        # Show prediction for current input if available
        if predict_button and scaler is not None:
            st.markdown("---")
            st.markdown("### 🎯 Best Model Prediction for Your Input")
            
            input_data_scaled = scaler.transform([input_data])
            prediction_best = best_model.predict(input_data_scaled)
            
            try:
                probability_best = best_model.predict_proba(input_data_scaled)
                if prediction_best[0] == 1:
                    st.error(f"**Prediction:** Heart Disease Risk Detected")
                    st.metric("Risk Probability", f"{probability_best[0][1]:.1%}")
                else:
                    st.success(f"**Prediction:** No Heart Disease Risk")
                    st.metric("Risk Probability", f"{probability_best[0][0]:.1%}")
            except AttributeError:
                if prediction_best[0] == 1:
                    st.error("**Prediction:** Heart Disease Risk Detected (Probability not available)")
                else:
                    st.success("**Prediction:** No Heart Disease Risk (Probability not available)")
    else:
        st.warning("Dataset not available for best model analysis.")

with tab7:
    st.subheader("High Risk Guide")
    
    st.markdown("""
    ### 🚨 Heart Disease Risk Management Guide
    
    This guide provides essential information for individuals at high risk of heart disease.
    """)
    
    # Emergency Information
    st.markdown("### 🚑 Emergency Information")
    st.error("""
    **If you experience any of these symptoms, call emergency services immediately:**
    - Chest pain or pressure
    - Shortness of breath
    - Pain radiating to arm, neck, or jaw
    - Nausea, lightheadedness, or cold sweats
    - Unusual fatigue
    """)
    
    # Risk Factors
    st.markdown("### ⚠️ Major Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Modifiable Risk Factors:**
        - 🚬 Smoking
        - 🍔 Poor diet
        - 🏃‍♂️ Physical inactivity
        - 🍷 Excessive alcohol
        - 😰 Stress
        - 📈 High blood pressure
        - 🩸 High cholesterol
        - 🍯 Diabetes
        - ⚖️ Obesity
        """)
    
    with col2:
        st.markdown("""
        **Non-Modifiable Risk Factors:**
        - 👴 Age (65+)
        - 👨‍👩‍👧‍👦 Family history
        - 🧬 Genetics
        - 👤 Gender (men at higher risk)
        - 🏥 Previous heart conditions
        """)
    
    # Lifestyle Changes
    st.markdown("### 💪 Lifestyle Changes")
    
    st.markdown("#### 🥗 Diet Recommendations")
    st.success("""
    **Heart-Healthy Diet:**
    - 🥬 Increase fruits and vegetables
    - 🐟 Eat fatty fish (salmon, mackerel)
    - 🥜 Include nuts and seeds
    - 🌾 Choose whole grains
    - 🫒 Use olive oil instead of butter
    - 🧂 Reduce salt intake
    - 🍬 Limit added sugars
    - 🥩 Choose lean proteins
    """)
    
    st.markdown("#### 🏃‍♂️ Exercise Guidelines")
    st.info("""
    **Physical Activity:**
    - 🚶‍♂️ 150 minutes moderate exercise/week
    - 🏃‍♂️ 75 minutes vigorous exercise/week
    - 💪 Strength training 2-3 times/week
    - 🧘‍♀️ Include flexibility exercises
    - 🚴‍♂️ Start slowly and build gradually
    """)
    
    # Medical Management
    st.markdown("### 🏥 Medical Management")
    
    st.markdown("#### 📋 Regular Check-ups")
    st.warning("""
    **Essential Medical Monitoring:**
    - 🩺 Annual physical examination
    - 💓 Regular blood pressure monitoring
    - 🩸 Cholesterol level checks
    - 🍯 Blood sugar monitoring (if diabetic)
    - ❤️ Cardiac stress tests (as recommended)
    - 📊 Weight and BMI tracking
    """)
    
    st.markdown("#### 💊 Medication Management")
    st.info("""
    **Common Medications:**
    - 💊 Blood pressure medications
    - 🩸 Cholesterol-lowering drugs
    - 💓 Blood thinners (if prescribed)
    - 🍯 Diabetes medications (if needed)
    - 💊 Aspirin therapy (consult doctor)
    """)
    
    # Stress Management
    st.markdown("### 😌 Stress Management")
    
    st.markdown("#### 🧘‍♀️ Relaxation Techniques")
    st.success("""
    **Stress Reduction Methods:**
    - 🧘‍♀️ Meditation and mindfulness
    - 🧘‍♂️ Deep breathing exercises
    - 🎵 Music therapy
    - 🌳 Nature walks
    - 📚 Reading and hobbies
    - 👥 Social support groups
    - 😴 Quality sleep (7-9 hours)
    """)
    
    # Warning Signs
    st.markdown("### ⚠️ Warning Signs to Watch")
    
    st.error("""
    **Seek Medical Attention If You Experience:**
    - 💔 Chest discomfort or pain
    - 😮‍💨 Shortness of breath
    - 💪 Arm, neck, or jaw pain
    - 🤢 Nausea or indigestion
    - 💨 Cold sweats
    - 😵 Dizziness or lightheadedness
    - 😴 Unusual fatigue
    - 💓 Irregular heartbeat
    - 🦵 Swelling in legs/ankles
    """)
    
    # Prevention Tips
    st.markdown("### 🛡️ Prevention Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Daily Habits:**
        - 🚭 Quit smoking
        - 🥗 Eat heart-healthy foods
        - 🏃‍♂️ Exercise regularly
        - 😴 Get adequate sleep
        - 🧘‍♀️ Manage stress
        - 🍷 Limit alcohol
        - 💊 Take medications as prescribed
        """)
    
    with col2:
        st.markdown("""
        **Long-term Goals:**
        - 📉 Maintain healthy weight
        - 🩸 Control blood pressure
        - 🍯 Manage diabetes
        - 🩸 Keep cholesterol in check
        - 🏥 Regular medical check-ups
        - 📚 Stay informed about heart health
        - 👥 Build support network
        """)
    
    # Resources
    st.markdown("### 📚 Additional Resources")
    
    st.markdown("""
    **Helpful Resources:**
    - 🏥 American Heart Association
    - 🩺 Centers for Disease Control (CDC)
    - 💓 National Heart, Lung, and Blood Institute
    - 🏥 Local cardiac rehabilitation programs
    - 👥 Heart disease support groups
    - 📱 Heart health mobile apps
    - 📚 Educational materials from your healthcare provider
    """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Important Disclaimer:**
    This guide is for informational purposes only and should not replace professional medical advice. 
    Always consult with your healthcare provider for personalized recommendations and treatment plans.
    """)

st.markdown("---")
