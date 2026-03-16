# Playwright 웹크롤링 가이드

> 학습 날짜: <!-- 직접 기입 -->

## 기본 설정
```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto("https://example.com")
    content = await page.content()
    await browser.close()
```

## Rate Limit 적용
```python
import asyncio, random

await asyncio.sleep(random.uniform(1, 3))
```

## 학습 포인트
<!-- 실습 후 채워넣기 -->
-
-
