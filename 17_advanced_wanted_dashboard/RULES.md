# RULES.md — 코딩 컨벤션 & 제약

## TypeScript (Frontend)
- `strict: true` 필수 — `any` 사용 금지
- 컴포넌트: 함수형 + Arrow function 통일
- Props 타입: `interface` 정의 후 사용 (inline 금지)
- API 호출: `src/api/` 모듈에서만 — 컴포넌트에서 직접 fetch 금지
- 상태 관리: 컴포넌트 로컬 `useState` 우선, 공유 상태는 Context API

## FastAPI (Backend)
- 모든 라우터 함수: `async def` 필수
- Pydantic 모델: `src/backend/models/schemas.py` 에서만 정의
- 의존성 주입: `Depends()` 활용 (DB 세션, 설정 등)
- 에러 응답: `HTTPException` + 구체적 status_code
- 환경변수: `pydantic-settings` BaseSettings 사용 (하드코딩 금지)

## 크롤링 (Playwright)
- Rate limit: 요청 간 **1~3초 랜덤 딜레이** 필수
- User-Agent: 고정값 사용 금지, 풀에서 랜덤 선택
- robots.txt 준수: 크롤링 전 허용 경로 확인
- 오류 처리: 개별 공고 실패 시 전체 중단하지 말고 skip + 로깅
- 데이터 저장: 크롤링 결과는 항상 타임스탬프와 함께 저장

## Claude API
- 토큰 비용 의식: JD 전문 대신 주요업무/자격요건 섹션만 전달
- 응답 포맷: `response_format={"type": "json_object"}` 명시
- 재시도: 지수 백오프 (최대 3회)
- 시스템 프롬프트: 200토큰 이내로 유지

## 공통
- `print()` 금지 — 모든 로깅은 `logging` 모듈
- 시크릿 (API 키 등): `.env` 파일 + `.gitignore` 처리
- 모듈 상단: `logger = logging.getLogger(__name__)`
