import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "gpt-4o-mini"
RESPONSES_URL = "https://api.openai.com/v1/responses"
ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "reports"
QA_DIR = ROOT_DIR / "qa"
SUMMARY_PATH = ROOT_DIR / "qa_summary.json"
ENV_PATH = Path(r"C:\Users\user\HANBEEN\_keys\.env.txt")
MAX_CONTEXT_CHARS = 120000
QUESTIONS_PER_COMPANY = 50
MAX_RETRIES = 3

KEYWORDS = [
    "매출",
    "영업이익",
    "당기순이익",
    "순이익",
    "매출액",
    "수익",
    "비용",
    "원가",
    "부채",
    "부채비율",
    "자산",
    "현금흐름",
    "투자",
    "차입",
    "리스크",
    "위험",
    "전망",
    "계획",
    "실적",
    "재무",
    "손익",
    "유동",
    "차입금",
    "CAPEX",
    "설비투자",
    "배당",
]


def load_openai_api_key() -> str:
    if not ENV_PATH.exists():
        raise RuntimeError(f"환경 파일이 없습니다: {ENV_PATH}")

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise RuntimeError("OPENAI_API_KEY를 _keys/.env.txt에서 찾지 못했습니다.")


def score_line(line: str) -> int:
    score = 0
    for keyword in KEYWORDS:
        if keyword in line:
            score += 3
    if re.search(r"\d", line):
        score += 1
    if len(line) > 140:
        score += 1
    return score


def build_context(report_text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in report_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    selected = []
    seen = set()

    # Keep metadata header.
    for line in lines[:40]:
        if line not in seen:
            selected.append(line)
            seen.add(line)

    scored_indexes = []
    for idx, line in enumerate(lines):
        score = score_line(line)
        if score > 0:
            scored_indexes.append((score, idx))

    scored_indexes.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    for _, idx in scored_indexes:
        start = max(0, idx - 3)
        end = min(len(lines), idx + 6)
        for line in lines[start:end]:
            if line not in seen:
                selected.append(line)
                seen.add(line)
        if sum(len(line) + 1 for line in selected) >= MAX_CONTEXT_CHARS:
            break

    context = "\n".join(selected)
    return context[:MAX_CONTEXT_CHARS]


def extract_response_text(payload: dict) -> str:
    if payload.get("output_text"):
        return payload["output_text"]

    parts = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts)


def call_openai(api_key: str, corp_name: str, context: str) -> list[dict]:
    prompt = (
        "아래는 기업 사업보고서 본문 일부이다. "
        "재무 분석가가 물어볼 법한 질문과 보고서 기반 답변을 50쌍 생성해라. "
        "매출, 영업이익, 순이익, 부채비율, 자금조달, 투자, 리스크, 전망 등 재무 핵심 항목 위주로 구성해라. "
        "답변은 제공된 본문에 근거한 내용만 사용하고, 본문에 없는 수치는 추정하지 마라. "
        "JSON 객체 형식으로만 출력해라. 최상위 키는 items 하나만 사용하고, "
        "items는 question, answer 문자열만 가진 50개 배열이어야 한다.\n\n"
        f"기업명: {corp_name}\n"
        "사업보고서 본문:\n"
        f"{context}"
    )

    schema = {
        "type": "json_schema",
        "name": "qa_pairs",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": QUESTIONS_PER_COMPANY,
                    "maxItems": QUESTIONS_PER_COMPANY,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                        "required": ["question", "answer"],
                    },
                }
            },
            "required": ["items"],
        },
        "strict": True,
    }

    body = {
        "model": MODEL,
        "input": prompt,
        "text": {"format": schema},
        "max_output_tokens": 12000,
    }

    request = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "ignore")
            raise RuntimeError(f"OpenAI API HTTP {exc.code}: {error_body}") from exc
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"OpenAI API 호출 실패: {exc}") from exc
            time.sleep(3 * attempt)

    text = extract_response_text(payload).strip()
    if not text:
        raise RuntimeError("OpenAI 응답에서 텍스트를 찾지 못했습니다.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON 파싱 실패: {exc}: {text[:500]}") from exc

    qa_pairs = parsed.get("items")
    if not isinstance(qa_pairs, list) or len(qa_pairs) != QUESTIONS_PER_COMPANY:
        raise RuntimeError(f"예상한 {QUESTIONS_PER_COMPANY}개 Q&A가 아니라 {len(qa_pairs)}개를 받았습니다.")

    for item in qa_pairs:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            raise RuntimeError("Q&A 항목 구조가 올바르지 않습니다.")

    return qa_pairs


def main() -> int:
    api_key = load_openai_api_key()
    QA_DIR.mkdir(parents=True, exist_ok=True)

    report_files = sorted(REPORTS_DIR.glob("*.txt"))
    if not report_files:
        raise RuntimeError(f"입력 파일이 없습니다: {REPORTS_DIR}")

    selected_names = set(sys.argv[1:])
    if selected_names:
        report_files = [path for path in report_files if path.stem in selected_names]
        if not report_files:
            raise RuntimeError(f"선택한 기업 파일을 찾지 못했습니다: {sorted(selected_names)}")

    all_pairs = []
    summary_results = []
    failures = []

    for report_path in report_files:
        corp_name = report_path.stem
        try:
            report_text = report_path.read_text(encoding="utf-8")
            context = build_context(report_text)
            if not context:
                raise RuntimeError("문맥 추출 결과가 비어 있습니다.")

            qa_pairs = call_openai(api_key, corp_name, context)

            company_payload = {
                "corp_name": corp_name,
                "source_file": str(report_path),
                "qa_count": len(qa_pairs),
                "items": qa_pairs,
            }
            company_path = QA_DIR / f"{corp_name}_qa.json"
            company_path.write_text(json.dumps(company_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            for item in qa_pairs:
                all_pairs.append(
                    {
                        "corp_name": corp_name,
                        "question": item["question"],
                        "answer": item["answer"],
                    }
                )

            summary_results.append(
                {
                    "corp_name": corp_name,
                    "source_file": str(report_path),
                    "qa_file": str(company_path),
                    "qa_count": len(qa_pairs),
                    "context_chars": len(context),
                }
            )
            print(f"[OK] {corp_name}: {len(qa_pairs)} pairs")
            time.sleep(1)
        except Exception as exc:
            failures.append({"corp_name": corp_name, "error": str(exc)})
            print(f"[ERROR] {corp_name}: {exc}", file=sys.stderr)

    all_pairs = []
    for company_file in sorted(QA_DIR.glob("*_qa.json")):
        if company_file.name == "all_qa.json":
            continue
        company_payload = json.loads(company_file.read_text(encoding="utf-8"))
        corp_name = company_payload.get("corp_name", company_file.stem.removesuffix("_qa"))
        for item in company_payload.get("items", []):
            all_pairs.append(
                {
                    "corp_name": corp_name,
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )

    all_path = QA_DIR / "all_qa.json"
    all_path.write_text(json.dumps(all_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    final_results = []
    for company_file in sorted(QA_DIR.glob("*_qa.json")):
        if company_file.name == "all_qa.json":
            continue
        company_payload = json.loads(company_file.read_text(encoding="utf-8"))
        final_results.append(
            {
                "corp_name": company_payload.get("corp_name", company_file.stem.removesuffix("_qa")),
                "source_file": company_payload.get("source_file", ""),
                "qa_file": str(company_file),
                "qa_count": len(company_payload.get("items", [])),
            }
        )

    summary = {
        "model": MODEL,
        "results": final_results,
        "failures": failures,
        "total_pairs": len(all_pairs),
        "all_qa_path": str(all_path),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print(f"saved_files: {len(summary_results)}")
    print(f"failed_files: {len(failures)}")
    print(f"total_pairs: {len(all_pairs)}")
    print(f"all_qa_path: {all_path}")
    print(f"summary_path: {SUMMARY_PATH}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
