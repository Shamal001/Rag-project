import sys
sys.path.append("/home/sk/rag-v2")  # query_data.py 경로 추가
from query_data import query_rag
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_monopoly_rules():
    print("✅ test_monopoly_rules 실행")
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )

def test_ticket_to_ride_rules():
    print("✅ test_ticket_to_ride_rules 실행")
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )

def query_and_validate(question: str, expected_response: str):
    print(f"🔍 질문: {question}")
    
    response_text = query_rag(question)
    print(f"📝 RAG 응답: {response_text}")

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    print(f"📜 생성된 프롬프트:\n{prompt}")

    model = Ollama(model="mistral")
    print("🚀 Ollama 모델 호출 시작")

    evaluation_results_str = model.predict(prompt)  # invoke → predict 변경
    print(f"📊 Ollama 응답: {evaluation_results_str}")

    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"✅ Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"❌ Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"❗ Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

# 테스트 실행
test_monopoly_rules()
test_ticket_to_ride_rules()
