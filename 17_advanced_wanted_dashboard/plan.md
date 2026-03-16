# Wanted Dashboard — 개발 체크리스트

## Phase 1: 백엔드 기반 구축
- [ ] `src/backend/pyproject.toml` 작성 (fastapi, playwright, anthropic, pydantic-settings)
- [ ] `models/schemas.py` — Job, InsightRequest, InsightResponse Pydantic 모델
- [ ] `main.py` — FastAPI 앱 초기화 + CORS 설정
- [ ] `GET /health` 헬스체크 엔드포인트

## Phase 2: 크롤러 구현
- [ ] `services/crawler.py` — Playwright 기반 wanted.co.kr 크롤러
  - [ ] 포지션/도메인 검색 URL 파라미터 확인
  - [ ] 공고 목록 파싱 (제목, 회사, 연차, 스킬)
  - [ ] Rate limit (1~3초 딜레이) 적용
  - [ ] 크롤링 결과 JSON 저장
- [ ] `routers/jobs.py` — GET /jobs, POST /jobs/crawl 구현
- [ ] 단독 실행 테스트: `python -m services.crawler`

## Phase 3: FastAPI → React 연결
- [ ] `src/frontend/` Vite + React + TypeScript 초기화
  ```bash
  npm create vite@latest frontend -- --template react-ts
  ```
- [ ] `src/api/jobs.ts` — /jobs API 호출 모듈
- [ ] `FilterPanel` 컴포넌트 — 포지션/도메인/연차 필터
- [ ] `JobCard` 컴포넌트 — 공고 카드 + 원티드 링크
- [ ] `JobBoard` 컴포넌트 — 필터 + 카드 리스트 통합
- [ ] CORS 동작 확인: frontend:5173 → backend:8000

## Phase 4: LLM 연동
- [ ] `.env` — `ANTHROPIC_API_KEY` 설정
- [ ] `services/llm.py` — Claude API 클라이언트 초기화
- [ ] `services/analyzer.py` — JD 분석 파이프라인
  - [ ] JD 핵심 섹션 추출 (토큰 절약)
  - [ ] Claude API 호출 + JSON 응답 파싱
  - [ ] 스킬 갭 분석 + 커리어 방향성 제안
- [ ] `routers/insights.py` — POST /insights/analyze
- [ ] `AIInsight` 컴포넌트 — LLM 분석 결과 패널

## Phase 5: 통합 & 배포
- [ ] `docker-compose.yml` — frontend + backend 동시 실행
- [ ] `docker-compose up` 정상 동작 확인
- [ ] API 통합 테스트
  - [ ] GET /jobs?position=ML Engineer 응답 확인
  - [ ] POST /insights/analyze Claude 응답 확인
- [ ] 스크린샷 → `screenshots/`
- [ ] 보고서 작성 → `reports/`
