# FastAPI + React 아키텍처 가이드

> 학습 날짜: <!-- 직접 기입 -->

## CORS 설정 (FastAPI)
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## React API 호출 패턴
```typescript
// src/api/jobs.ts
export async function fetchJobs(filters: JobFilters): Promise<Job[]> {
  const params = new URLSearchParams(filters as Record<string, string>);
  const res = await fetch(`http://localhost:8000/jobs?${params}`);
  if (!res.ok) throw new Error("API 오류");
  return res.json();
}
```

## Docker Compose 연결
```yaml
services:
  backend:
    build: ./src/backend
    ports: ["8000:8000"]
  frontend:
    build: ./src/frontend
    ports: ["3000:80"]
    environment:
      - VITE_API_URL=http://backend:8000
```

## 학습 포인트
<!-- 실습 후 채워넣기 -->
-
-
