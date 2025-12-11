import streamlit as st
import httpx
import json
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import from app package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.json_cleaner import clean_json

API_URL = "http://localhost:8000/evaluate"   # Change when deployed

st.set_page_config(page_title="EvalFlow — LLM Evaluation", layout="wide")
st.title("🚀 EvalFlow — LLM Evaluation Pipeline Tester")

st.write(
    "Upload your **conversation JSON** and **context JSON**, then click *Evaluate* "
    "to generate the reliability report."
)

# ---- Sidebar: API Configuration ----
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("### API Keys")
st.sidebar.info("💡 Leave fields empty to use keys from .env file")

# Initialize session state for API keys
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""
if "google_key" not in st.session_state:
    st.session_state.google_key = ""
if "api_configured" not in st.session_state:
    st.session_state.api_configured = False

# OpenAI API Key
openai_key = st.sidebar.text_input(
    "🔑 OpenAI API Key",
    value=st.session_state.openai_key,
    type="password",
    help="(Optional) Enter to override .env value. Leave empty to use .env"
)
if openai_key != st.session_state.openai_key:
    st.session_state.openai_key = openai_key

# Google API Key
google_key = st.sidebar.text_input(
    "🔑 Google API Key",
    value=st.session_state.google_key,
    type="password",
    help="(Optional) Enter to override .env value. Leave empty to use .env"
)
if google_key != st.session_state.google_key:
    st.session_state.google_key = google_key

# Model Selection
st.sidebar.markdown("### Model Configuration")
provider = st.sidebar.radio(
    "Select LLM Provider",
    ("OpenAI", "Google Gemini"),
    help="Choose which LLM provider to use for evaluation"
)

# Model selection based on provider
if provider == "OpenAI":
    openai_models = {
        "GPT-4o Mini (Fast & Cheap)": "gpt-4o-mini",
        "GPT-4 Turbo (Most Capable)": "gpt-4-turbo",
        "GPT-3.5 Turbo (Legacy)": "gpt-3.5-turbo"
    }
    selected_model_display = st.sidebar.selectbox(
        "📊 Select OpenAI Model",
        list(openai_models.keys()),
        help="Free tier supports GPT-4o Mini (limited requests)"
    )
    selected_model = openai_models[selected_model_display]
    st.sidebar.caption("💡 Free tier: Use GPT-4o Mini for best value")
else:
    gemini_models = {
        "Gemini 2.5 Flash (Latest)": "gemini-2.5-flash",
        "Gemini 2.0 Flash (Fast)": "gemini-2.0-flash",
        "Gemini 1.5 Flash (Recommended)": "gemini-1.5-flash",
        "Gemini 1.5 Pro (Most Capable)": "gemini-1.5-pro"
    }
    selected_model_display = st.sidebar.selectbox(
        "📊 Select Gemini Model",
        list(gemini_models.keys()),
        help="Free tier supports all models with quota limits"
    )
    selected_model = gemini_models[selected_model_display]
    st.sidebar.caption("💡 Free tier: Generous quota for all models")

# Apply Configuration Button
if st.sidebar.button("✅ Apply Configuration", use_container_width=True):
    st.session_state.api_configured = True
    st.sidebar.success("✅ Configuration applied!")
    
    # Display which keys will be used
    if openai_key:
        st.sidebar.caption("📝 Using OpenAI key from UI input")
    else:
        st.sidebar.caption("📝 Will use OpenAI key from .env")
    
    if google_key:
        st.sidebar.caption("📝 Using Google key from UI input")
    else:
        st.sidebar.caption("📝 Will use Google key from .env")

# Show configuration status
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
if st.session_state.api_configured:
    st.sidebar.success("✅ UI Keys Configured")
    st.sidebar.caption("Ready to evaluate with provided keys")
else:
    st.sidebar.info("ℹ️ Using .env keys")
    st.sidebar.caption("Click 'Apply Configuration' to override with UI keys")

# ---- File Upload Section ----
st.markdown("---")
st.markdown("### 📤 Input Files")
conv_file = st.file_uploader("📄 Upload Conversation JSON", type=["json"], key="conv_uploader")
ctx_file = st.file_uploader("📄 Upload Context JSON", type=["json"], key="ctx_uploader")

# ---- Evaluation Action ----
col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("▶️ Run Evaluation", use_container_width=True)
    
with col2:
    st.empty()

if run_button:
    # Validate files exist
    if not conv_file or not ctx_file:
        st.error("❌ Please upload both conversation.json and context.json")
        st.stop()

    # Create a container for errors that won't re-render
    error_container = st.container()
    
    try:
        # Parse uploaded JSONs with cleaning
        conv_raw = conv_file.read().decode("utf-8")
        ctx_raw = ctx_file.read().decode("utf-8")
        
        # Validate files are not empty
        if not conv_raw.strip():
            with error_container:
                st.error("❌ Conversation file is empty. Please upload a valid JSON file.")
            st.stop()
        if not ctx_raw.strip():
            with error_container:
                st.error("❌ Context file is empty. Please upload a valid JSON file.")
            st.stop()
        
        # Clean and parse JSON
        conversation_json = json.loads(clean_json(conv_raw))
        context_json = json.loads(clean_json(ctx_raw))
    except ValueError as e:
        with error_container:
            st.error(f"⚠️ JSON Validation Error: {str(e)}")
            st.info("💡 Tip: Make sure your JSON files are not empty and contain valid JSON data.")
        st.stop()
    except Exception as e:
        with error_container:
            st.error(f"⚠️ Error parsing JSON: {type(e).__name__}: {str(e)}")
            st.info("💡 Tip: Ensure files are valid JSON format (not binary or corrupted).")
        st.stop()

    payload = {
        "conversation": conversation_json,
        "context_vectors": context_json,
        "model_type": "openai" if provider == "OpenAI" else "gemini",
        "model_name": selected_model
    }

    # ---- API Call ----
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="🔄 Starting evaluation...")
        status_text = st.empty()
        
        try:
            # Step 1: Parse files
            status_text.text("📂 Parsing JSON files...")
            progress_bar.progress(10, text="📂 Parsing JSON files... (10%)")
            
            client = httpx.Client(timeout=50.0)   # long timeout for LLM processing
            
            # Step 2: Send request
            status_text.text("📡 Sending request to API...")
            progress_bar.progress(20, text="📡 Sending request to API... (20%)")
            
            response = client.post(API_URL, json=payload)
            client.close()
            
            # Step 3: Check response
            status_text.text("✔️ Received response from API...")
            progress_bar.progress(30, text="✔️ Received response from API... (30%)")
            
            if response.status_code != 200:
                st.error(f"❌ API Error [{response.status_code}]: {response.text}")
                st.stop()

            result = response.json()
            
            # Step 4: Processing results
            status_text.text("🔄 Processing results...")
            progress_bar.progress(50, text="🔄 Processing results... (50%)")
            
            # Step 5: Rendering report
            status_text.text("🎨 Rendering evaluation report...")
            progress_bar.progress(75, text="🎨 Rendering evaluation report... (75%)")

            st.success("✅ Evaluation Completed Successfully!")
            progress_bar.progress(100, text="✅ Complete! (100%)")
            
            # Clear status after completion
            status_text.empty()
            
            # ---- Display Results in Tabular Format ----
            st.subheader("📊 Evaluation Report")
            
            # 1. Scores Summary (Table)
            st.markdown("### 📈 Evaluation Scores")
            scores_df = pd.DataFrame({
                "Metric": ["Relevance", "Completeness", "Accuracy"],
                "Score (1-10)": [
                    result["relevance_score"],
                    result["completeness_score"],
                    result["accuracy_score"]
                ]
            })
            st.dataframe(scores_df, width='stretch')
            
            # 2. Generated Response
            st.markdown("### 💬 Generated Response")
            st.info(result["generated_response"])
            
            # 3. Prompt Used
            st.markdown("### 🔤 Prompt Used for Generation")
            with st.expander("View Prompt"):
                st.text(result["prompt_used"])
            
            # 4. Performance Metrics
            st.markdown("### ⏱️ Performance Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Latency", "Cost"],
                "Value": [
                    f"{result['latency_ms']:.2f} ms",
                    f"${result['cost_usd']:.4f}"
                ]
            })
            st.dataframe(metrics_df, width='stretch')
            
            # 5. Hallucinations
            st.markdown("### ⚠️ Detected Hallucinations")
            if result["hallucinations"]:
                hal_df = pd.DataFrame({
                    "Hallucination": result["hallucinations"]
                })
                st.dataframe(hal_df, width='stretch')
            else:
                st.success("✅ No hallucinations detected")
            
            # 6. Retrieved Context with Similarity Scores
            st.markdown("### 🔍 Retrieved Context (Ranked by Similarity)")
            if result["retrieved_context"]:
                context_data = []
                for i, ctx in enumerate(result["retrieved_context"], 1):
                    context_data.append({
                        "#": i,
                        "Similarity Score": f"{ctx['similarity_score']:.4f}",
                        "Source URL": ctx['source_url'] if ctx['source_url'] else "N/A",
                        "Context Text": ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
                    })
                context_df = pd.DataFrame(context_data)
                st.dataframe(context_df, width='stretch', height=400)
                
                # Option to expand and view full context
                st.markdown("#### Full Context Details")
                for i, ctx in enumerate(result["retrieved_context"], 1):
                    with st.expander(f"Context #{i} (Similarity: {ctx['similarity_score']:.4f})"):
                        st.write(f"**Source URL:** {ctx['source_url'] if ctx['source_url'] else 'N/A'}")
                        st.write(f"**Text:** {ctx['text']}")
            else:
                st.warning("⚠️ No context retrieved")
            
            # 7. Explanations
            st.markdown("### 📝 Evaluation Explanations")
            if result["explanations"]:
                for key, explanation in result["explanations"].items():
                    with st.expander(f"📄 {key.replace('_', ' ').title()}"):
                        st.write(explanation)
            
            # 8. Raw JSON (for reference)
            st.markdown("### 📋 Raw JSON Response")
            with st.expander("View Raw JSON"):
                st.json(result)

        except Exception as e:
            st.error(f"🚨 Error connecting to backend: {e}")
            progress_bar.progress(0, text="❌ Error occurred")

    if not conv_file or not ctx_file:
        st.error("❌ Please upload both conversation.json and context.json")
        st.stop()

    try:
        # Parse uploaded JSONs with cleaning
        conv_raw = conv_file.read().decode("utf-8")
        ctx_raw = ctx_file.read().decode("utf-8")
        
        conversation_json = json.loads(clean_json(conv_raw))
        context_json = json.loads(clean_json(ctx_raw))
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error parsing JSON: {str(e)}")
        st.stop()

    payload = {
        "conversation": conversation_json,
        "context_vectors": context_json
    }

    # ---- API Call ----
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0, text="🔄 Starting evaluation...")
        status_text = st.empty()
        
        try:
            # Step 1: Parse files
            status_text.text("📂 Parsing JSON files...")
            progress_bar.progress(10, text="📂 Parsing JSON files... (10%)")
            
            client = httpx.Client(timeout=50.0)   # long timeout for LLM processing
            
            # Step 2: Send request
            status_text.text("📡 Sending request to API...")
            progress_bar.progress(20, text="📡 Sending request to API... (20%)")
            
            response = client.post(API_URL, json=payload)
            client.close()
            
            # Step 3: Check response
            status_text.text("✔️ Received response from API...")
            progress_bar.progress(30, text="✔️ Received response from API... (30%)")
            
            if response.status_code != 200:
                st.error(f"❌ API Error [{response.status_code}]: {response.text}")
                st.stop()

            result = response.json()
            
            # Step 4: Processing results
            status_text.text("🔄 Processing results...")
            progress_bar.progress(50, text="🔄 Processing results... (50%)")
            
            # Step 5: Rendering report
            status_text.text("🎨 Rendering evaluation report...")
            progress_bar.progress(75, text="🎨 Rendering evaluation report... (75%)")

            st.success("✅ Evaluation Completed Successfully!")
            progress_bar.progress(100, text="✅ Complete! (100%)")
            
            # Clear status after completion
            status_text.empty()
            
            # ---- Display Results in Tabular Format ----
            st.subheader("📊 Evaluation Report")
            
            # 1. Scores Summary (Table)
            st.markdown("### 📈 Evaluation Scores")
            scores_df = pd.DataFrame({
                "Metric": ["Relevance", "Completeness", "Accuracy"],
                "Score (1-10)": [
                    result["relevance_score"],
                    result["completeness_score"],
                    result["accuracy_score"]
                ]
            })
            st.dataframe(scores_df, width='stretch')
            
            # 2. Generated Response
            st.markdown("### 💬 Generated Response")
            st.info(result["generated_response"])
            
            # 3. Prompt Used
            st.markdown("### 🔤 Prompt Used for Generation")
            with st.expander("View Prompt"):
                st.text(result["prompt_used"])
            
            # 4. Performance Metrics
            st.markdown("### ⏱️ Performance Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Latency", "Cost"],
                "Value": [
                    f"{result['latency_ms']:.2f} ms",
                    f"${result['cost_usd']:.4f}"
                ]
            })
            st.dataframe(metrics_df, width='stretch')
            
            # 5. Hallucinations
            st.markdown("### ⚠️ Detected Hallucinations")
            if result["hallucinations"]:
                hal_df = pd.DataFrame({
                    "Hallucination": result["hallucinations"]
                })
                st.dataframe(hal_df, width='stretch')
            else:
                st.success("✅ No hallucinations detected")
            
            # 6. Retrieved Context with Similarity Scores
            st.markdown("### 🔍 Retrieved Context (Ranked by Similarity)")
            if result["retrieved_context"]:
                context_data = []
                for i, ctx in enumerate(result["retrieved_context"], 1):
                    context_data.append({
                        "#": i,
                        "Similarity Score": f"{ctx['similarity_score']:.4f}",
                        "Source URL": ctx['source_url'] if ctx['source_url'] else "N/A",
                        "Context Text": ctx['text'][:100] + "..." if len(ctx['text']) > 100 else ctx['text']
                    })
                context_df = pd.DataFrame(context_data)
                st.dataframe(context_df, width='stretch', height=400)
                
                # Option to expand and view full context
                st.markdown("#### Full Context Details")
                for i, ctx in enumerate(result["retrieved_context"], 1):
                    with st.expander(f"Context #{i} (Similarity: {ctx['similarity_score']:.4f})"):
                        st.write(f"**Source URL:** {ctx['source_url'] if ctx['source_url'] else 'N/A'}")
                        st.write(f"**Text:** {ctx['text']}")
            else:
                st.warning("⚠️ No context retrieved")
            
            # 7. Explanations
            st.markdown("### 📝 Evaluation Explanations")
            if result["explanations"]:
                for key, explanation in result["explanations"].items():
                    with st.expander(f"📄 {key.replace('_', ' ').title()}"):
                        st.write(explanation)
            
            # 8. Raw JSON (for reference)
            st.markdown("### 📋 Raw JSON Response")
            with st.expander("View Raw JSON"):
                st.json(result)

        except Exception as e:
            st.error(f"🚨 Error connecting to backend: {e}")
            progress_bar.progress(0, text="❌ Error occurred")