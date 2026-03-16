# Claude API 연동 가이드

> 학습 날짜: <!-- 직접 기입 -->

## 기본 호출
```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "JD 분석 요청..."}
    ]
)
print(message.content[0].text)
```

## JSON 응답 강제
```python
# 시스템 프롬프트에 명시
system = "항상 유효한 JSON만 응답하세요."

# 응답 파싱
import json
result = json.loads(message.content[0].text)
```

## 토큰 절약 전략
- JD 전문 대신 주요업무 + 자격요건 섹션만 전달
- 시스템 프롬프트 200토큰 이내 유지
- 배치 가능하면 단일 요청으로 묶기

## 학습 포인트
<!-- 실습 후 채워넣기 -->
-
-
