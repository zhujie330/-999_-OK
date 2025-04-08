import streamlit as st

st.set_page_config(page_title="图像视频检测", layout="wide")


st.sidebar.header("🏠 Home")
st.write("# WebDemo for Deepfake Detection and Analysis! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
        """
        This Website is a Demo for Detection and Analysis of Deepfake Face in image and video .
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### What is Deepfake?
        Deepfake technology is a method that uses deep learning and artificial intelligence to synthesize fake images, audio, or videos. By synthesizing facial features of one person onto another person's image, deepfake technology can create very realistic-looking fake content.
        ### Main Techniques of Deepfake
        Deepfake technology relies on several main techniques, including:
        - **Facial Synthesis**: Using deep learning models to synthesize facial features of one person onto another person's image or video.
        - **Voice Synthesis**: Generating realistic-sounding speech using deep learning models to make it sound like it's spoken by a specific person.
        - **Text Synthesis**: Generating text content that can be used to create realistic fake articles, comments, etc.

        ### Methods of Deepfake Detection
        
        Methods for detecting deepfakes are continually evolving, with some common techniques including:
        
        - **Manual Inspection**: Professionals carefully examine details in the imagery to judge if there are any signs of manipulation.
        - **Digital Signatures and Watermarks**: Embedding digital signatures or watermarks into original content to verify its authenticity.
        - **Detection Models**: Using deep learning models to identify features of deepfake content, such as unnatural distortions in facial regions.

        ### Want to learn more?
        - CNN-generated images are surprisingly easy to spot...for now [CNNDetection](https://github.com/peterwang512/CNNDetection)
        - Leveraging Frequency Analysis for Deep Fake Image Recognition [FreDect](https://github.com/RUB-SysSec/GANDCTAnalysis)
        - Fusing global and local features for generalized AI-synthesized image detection [Fusing](https://github.com/littlejuyan/FusingGlobalandLocal)
        - Global Texture Enhancement for Fake Face Detection In the Wild [Gram-Net](https://github.com/liuzhengzhe/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild)
        - Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection [LGrad](https://github.com/chuangchuangtan/LGrad)
        - Detecting Generated Images by Real Images [LNP](https://github.com/Tangsenghenshou/Detecting-Generated-Images-by-Real-Images)
          
    """
    )