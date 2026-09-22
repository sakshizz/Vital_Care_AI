# Import necessary libraries
import streamlit as st 
import os
import google.generativeai as genai
from PIL import Image
import time
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="VitalCare AI", 
    page_icon="🏥",
    layout="centered"
)

# Initialize session state for rate limiting
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = None
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0

def can_make_request():
    """Check if we can make a request based on rate limits"""
    now = datetime.now()
    
    # Reset counter every minute
    if (st.session_state.last_request_time is None or 
        now - st.session_state.last_request_time > timedelta(minutes=1)):
        st.session_state.request_count = 0
    
    # Conservative limit: 8 requests per minute
    if st.session_state.request_count >= 8:
        return False, "Rate limit reached. Please wait 1 minute before next analysis."
    
    return True, ""

def update_request_tracker():
    """Update request tracking"""
    st.session_state.last_request_time = datetime.now()
    st.session_state.request_count += 1

# Header
st.title("🏥 VitalCare AI")
st.subheader("AI-Powered Medical Image Analysis")
st.markdown("---")

# API Setup
api_key = os.getenv("API_KEY")
if not api_key:
    st.error("❌ API key not found! Please set API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Model setup
@st.cache_resource
def get_model():
    """Initialize the AI model"""
    try:
        # Try gemini-1.5-flash first (higher free tier limits)
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2048,
            }
        )
        return model
    except Exception as e:
        # Fallback to gemini-1.5-pro
        try:
            model = genai.GenerativeModel(
                model_name='gemini-1.5-pro',
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2048,
                }
            )
            return model
        except Exception as e2:
            st.error(f"❌ Failed to initialize AI model: {e2}")
            return None

model = get_model()
if model is None:
    st.stop()

# Medical analysis prompt
MEDICAL_PROMPT = """
You are an expert medical image analysis assistant. Please analyze this medical image and provide a structured report with the following sections:

## IMAGE TYPE & QUALITY
- Identify the type of medical imaging (X-ray, CT, MRI, ultrasound, etc.)
- Comment on image quality and visibility

## ANATOMICAL STRUCTURES
- Describe the main anatomical structures visible
- Note their appearance and positioning

## OBSERVATIONS
- List any notable findings, abnormalities, or areas of concern
- Describe any variations from normal appearance
- Comment on symmetry, density, or other relevant features

## CLINICAL RECOMMENDATIONS
- Suggest potential next steps or additional imaging if needed
- Recommend consultation with specific specialists if appropriate

## IMPORTANT LIMITATIONS
- Note any limitations in the analysis due to image quality or viewing angle
- Mention areas that require clinical correlation

Please provide me an output response with these 4 headings Detailed Analysis, Findings Report,  Recommendations and Next Steps, Treatment suggestions
"""

# File Upload Section
st.header("📸 Upload Medical Image")

uploaded_file = st.file_uploader(
    "Choose a medical image for analysis",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG • Maximum size: 5MB"
)

if uploaded_file is not None:
    # File validation
    file_size = len(uploaded_file.getvalue())
    max_size = 5 * 1024 * 1024  # 5MB
    
    if file_size > max_size:
        st.error(f"❌ File size ({file_size/1024/1024:.1f}MB) exceeds 5MB limit. Please use a smaller image.")
        st.stop()
    
    # Display uploaded image
    try:
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(image, caption=f"📋 {uploaded_file.name}", use_column_width=True)
        
        with col2:
            st.metric("Image Size", f"{image.size[0]} × {image.size[1]}")
            st.metric("File Size", f"{file_size/1024:.1f} KB")
            st.metric("Format", image.format)
            
    except Exception as e:
        st.error(f"❌ Cannot load image: {e}")
        st.stop()

    st.success("✅ Image uploaded successfully!")
    
    # Analysis Section
    st.header("🔬 Medical Analysis")
    
    # Rate limiting check
    can_request, limit_msg = can_make_request()
    
    if not can_request:
        st.warning(f"⏳ {limit_msg}")
        remaining_time = 60 - (datetime.now() - st.session_state.last_request_time).seconds
        st.info(f"⏰ Please wait {remaining_time} more seconds")
    else:
        # Analysis button
        if st.button("🔍 Analyze Medical Image", type="primary", use_container_width=True):
            
            with st.spinner("🔄 Analyzing medical image... This may take 30-60 seconds"):
                try:
                    # Update rate tracking
                    update_request_tracker()
                    
                    # Small delay to respect API limits
                    time.sleep(3)
                    
                    # Reset file pointer and prepare image
                    uploaded_file.seek(0)
                    pil_image = Image.open(uploaded_file)
                    
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Generate analysis
                    response = model.generate_content([MEDICAL_PROMPT, pil_image])
                    
                    if response and hasattr(response, 'text') and response.text:
                        # Display results
                        st.success("✅ Analysis completed successfully!")
                        
                        # Analysis report
                        st.markdown("## 📋 Medical Image Analysis Report")
                        st.markdown(response.text)
                        
                        # Generate report for download
                        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                        report_content = f"""MEDICAL IMAGE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Image: {uploaded_file.name}
Size: {image.size[0]} × {image.size[1]} pixels

{response.text}

---
DISCLAIMER: This AI-generated analysis is for educational and informational purposes only. 
It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
Always consult qualified healthcare professionals for medical decisions.
"""
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Analysis Report",
                            data=report_content,
                            file_name=f"medical_analysis_{timestamp}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    else:
                        st.error("❌ No analysis generated. Please try again.")
                        
                except Exception as e:
                    error_msg = str(e)
                    
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("❌ API rate limit exceeded!")
                        st.info("💡 Please wait 1-2 minutes before trying again.")
                        
                    elif "safety" in error_msg.lower():
                        st.error("❌ Image was flagged by safety filters.")
                        st.info("💡 Please try a different medical image.")
                        
                    else:
                        st.error(f"❌ Analysis failed: {error_msg}")
                        st.info("💡 Please try again or contact support if the issue persists.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    
    st.markdown("""
    ### 🎯 Supported Images
    - **X-rays** (chest, bone, dental)
    - **CT scans** (brain, chest, abdomen)
    - **MRI images** (brain, spine, joints)
    - **Ultrasounds** (abdominal, cardiac)
    - **Microscopy** (histology, pathology)
    
    ### 📋 Best Practices
    - Use clear, high-resolution images
    - Ensure proper contrast and lighting
    - Include relevant patient positioning
    - Remove any personal identifiers
    
    ### ⚠️ Important Notes
    - This tool is for educational purposes
    - Always consult medical professionals
    - AI analysis has limitations
    - Results require clinical correlation
    """)
    
    st.markdown("---")
    
    # Usage stats
    st.markdown("### 📊 Usage Stats")
    st.info(f"Requests this session: {st.session_state.request_count}")
    
    if st.session_state.last_request_time:
        last_request = st.session_state.last_request_time.strftime('%H:%M:%S')
        st.info(f"Last request: {last_request}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p><strong>⚠️ Medical Disclaimer</strong></p>
    <p>This AI tool is for educational and research purposes only. It does not provide medical advice, 
    diagnosis, or treatment recommendations. Always consult qualified healthcare professionals 
    for medical decisions and never rely solely on AI analysis for patient care.</p>
    <p><em>VitalCare AI • Powered by Google Gemini</em></p>
</div>
""", unsafe_allow_html=True)
