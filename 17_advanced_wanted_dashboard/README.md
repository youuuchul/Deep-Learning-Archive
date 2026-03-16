# Mission 17 (심화) — Wanted 채용 대시보드

원티드랩 포지션 포트폴리오용 채용 공고 대시보드.
Playwright 크롤링 + Claude API 분석 + React 프론트엔드.

## 기술 스택
- **Frontend**: React + TypeScript (Vite)
- **Backend**: FastAPI + Playwright
- **LLM**: Claude API (JD 분석)
- **배포**: Docker Compose

## 실행 방법
```bash
# 전체 실행
docker-compose up

# 개발 모드
cd src/backend && uvicorn main:app --reload --port 8000
cd src/frontend && npm install && npm run dev  # localhost:5173
```

## 폴더 구조
```
17_advanced_wanted_dashboard/
├── src/
│   ├── frontend/    — React 앱
│   └── backend/     — FastAPI + 크롤러 + LLM
├── CLAUDE.md        — 아키텍처 + API 스펙
├── RULES.md         — 코딩 컨벤션
└── plan.md          — 개발 체크리스트
```
