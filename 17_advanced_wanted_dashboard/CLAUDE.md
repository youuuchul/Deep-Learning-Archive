# CLAUDE.md — Wanted Dashboard (심화 미션)

## 프로젝트 개요
원티드랩 포지션 포트폴리오용 채용 공고 대시보드.
웹크롤링으로 채용 데이터를 수집하고, Claude API로 JD를 분석해 커리어 방향성을 제안.

## 아키텍처
```
React (Vite) ──── FastAPI ──── Playwright 크롤러
                      └──── Claude API (JD 분석)
```

## API 엔드포인트 스펙

### Jobs
| Method | Path | 설명 |
|--------|------|------|
| GET | `/jobs` | 채용 공고 목록 (필터: position, domain, experience) |
| GET | `/jobs/{id}` | 개별 공고 상세 |
| POST | `/jobs/crawl` | 크롤링 트리거 (백그라운드 태스크) |

### Insights
| Method | Path | 설명 |
|--------|------|------|
| POST | `/insights/analyze` | JD 분석 → 커리어 방향성 제안 |
| POST | `/insights/resume-fit` | 이력서 ↔ JD 매칭 점수 |

### 요청/응답 예시
```json
// GET /jobs?position=ML Engineer&domain=fintech
{
  "total": 12,
  "items": [
    {
      "id": "wanted-12345",
      "title": "ML Engineer",
      "company": "토스",
      "domain": "fintech",
      "experience": "3-5년",
      "url": "https://www.wanted.co.kr/wd/12345",
      "posted_at": "2026-03-15",
      "skills": ["Python", "PyTorch", "Kubernetes"]
    }
  ]
}

// POST /insights/analyze
// body: { "job_ids": ["wanted-12345"], "user_background": "..." }
{
  "summary": "...",
  "skill_gaps": [...],
  "recommendations": [...]
}
```

## LLM 프롬프트 전략
- 역할: JD 분석 전문가
- 컨텍스트: 수집된 JD 원문 (XML 구조화)
- 지시: 필수 스킬 추출 → 난이도 분류 → 커리어 단계별 우선순위
- 포맷: JSON 구조화 응답 (파싱 안정성)
- 토큰 절약: JD 전문 대신 핵심 섹션(주요업무, 자격요건)만 전달

## 프론트엔드 컴포넌트 역할
| 컴포넌트 | 역할 |
|---------|------|
| `JobBoard` | 전체 대시보드 레이아웃 + 상태 관리 |
| `FilterPanel` | 포지션 / 도메인 / 연차 필터 UI |
| `JobCard` | 개별 공고 카드 (링크 연결) |
| `AIInsight` | LLM 분석 결과 패널 |

## 실행 방법
```bash
# 전체 (docker-compose)
docker-compose up

# 백엔드만
cd src/backend && uvicorn main:app --reload

# 프론트만
cd src/frontend && npm install && npm run dev
```
