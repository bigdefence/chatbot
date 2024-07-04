import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 페이지 설정
st.set_page_config(page_title="Llama3 기반 한국어 챗봇", layout="wide")

# 사이드바 설정
st.sidebar.title("챗봇 정보")
st.sidebar.info(
    "이 챗봇은 Llama3 모델을 기반으로 한 한국어 대화 시스템입니다. "
    "다양한 주제에 대해 대화를 나눌 수 있습니다."
)

st.sidebar.title("사용 방법")
st.sidebar.markdown(
    """
    1. 채팅 입력창에 질문이나 대화를 입력하세요.
    2. 엔터를 누르면 AI가 응답합니다.
    3. 대화 기록은 자동으로 저장됩니다.
    """
)

st.sidebar.title("개발자 정보")
st.sidebar.markdown(
    """
    - **개발자**: 정강빈
    - **버전**: 1.0.0
    """
)

# 모델 로드 함수
@st.cache_resource
def load_model():
    model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

# 모델 로드
model, tokenizer = load_model()

# 응답 생성 함수
def generate_response(instruction):
    PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
    
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": instruction}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
    
    return tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

# 메인 앱
def main():
    st.title('Llama3 기반 한국어 챗봇')
    
    # 세션 상태 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # 사용자 입력
    if prompt := st.chat_input('질문을 입력하세요'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message('assistant'):
            with st.spinner('답변 생성 중...'):
                response = generate_response(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()