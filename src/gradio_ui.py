import gradio as gr
from chatbot import chatbot  # 기존 chatbot class 그대로 사용

with open("openai_key.txt", "r") as f:
    openai_key = f.read().strip()

bot = chatbot(openai_key)
bot.set_mode("D")  # 기본 모드 설정 (N: Normal, D: Discuss)
bot.set_main_book("The_Prince")  # 기본 책 제목 설정 (예시)

# 외부에서 mode 설정할 수 있게 변경 필요
def chat_interface(user_input, mode, history_state):
    bot.history = history_state or []
    if mode == "토론":
        bot.set_mode("D")
    else:
        bot.set_mode("N")

    response = bot.process_query(user_input)
    bot.update_history(user_input, response)

    # 히스토리를 Gradio용으로 변환 (list of tuples)
    chat_display = [(q, r) for q, r in bot.actual_history]

    return chat_display, bot.history

iface = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.Textbox(label="질문 입력"),
        gr.Radio(["토론", "일반"], label="모드 선택", value="토론"),
        gr.State([])
    ],
    outputs=[
        gr.Chatbot(label="대화"),
        gr.State()
    ],
    title="독서 토론 챗봇",
)

iface.launch()
