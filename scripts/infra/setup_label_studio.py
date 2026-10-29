import os

from label_studio_sdk import Client

# Connect to Label Studio
api_key = os.environ.get("LABEL_STUDIO_API_KEY")
if not api_key:
    raise SystemExit("Set LABEL_STUDIO_API_KEY before creating the project")
ls_url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8081")
ls = Client(url=ls_url, api_key=api_key)

# Create Myanmar ASR Project
project = ls.start_project(
    title="Myanmar ASR - Audio Tanscription Review",
    label_config="""
    <View style="display: flex; flex-direction: column; gap: 10px;">
        <Header value="🎙️ Myanmar Audio Transcription Review"/>

        <!-- Audio player with waveform -->
        <Audio name="audio" value="$audio"
                zoom="true" waveHeight="100"
                speed="true" volume="true"/>

        <!-- Show original auto-transcript -->
        <Header value="Original Transcript (auto):"/>
        <Text name="original_text" value="$sentence"
                style="background: #f0f0f0; padding: 10px; border-radius: 4px;"/>

        <!-- Editable corrected transcript -->
        <Header value="✏️ Corrected Transcript (Myanmar):"/>
        <TextArea name="corrected_text" toName="audio"
                    placeholder="ပြင်ဆင်ထားသော မြန်မာဘာသာ စာသားကို ဤနေရာတွင် ရိုက်ထည့်ပါ..."
                    rows="4" editable="true" maxSubmissions="1"/>

        <!-- Quality rating -->
        <Header value="Audio Quality:"/>
        <Choices name="quality" toName="audio"
                choice="single-radio" showInline="true">
            <Choice value="clean"   alias="🟢 Clean"/>
            <Choice value="noisy"   alias="🟡 Noisy"/>
            <Choice value="unclear" alias="🔴 Unclear"/>
            <Choice value="reject"  alias="⛔ Reject"/>
        </Choices>

        <!-- Source tag (read-only display) -->
        <Header value="Dataset Source:"/>
        <Text name="source_tag" value="$source"/>
        </View>
    """
)

print(f"✅ Project created: ID={project.id}")
print(f"   URL: {ls_url}/projects/{project.id}/")
