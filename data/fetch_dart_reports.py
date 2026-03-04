import io
import json
import re
import sys
import urllib.parse
import urllib.request
import warnings
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning


API_KEY = "8c3c01ba204c47510725364377a9945d69e6f44f"
LIST_API_URL = "https://opendart.fss.or.kr/api/list.json"
DOCUMENT_API_URL = "https://opendart.fss.or.kr/api/document.xml"
OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
SUMMARY_PATH = Path(__file__).resolve().parent / "reports_summary.json"

CORP_TARGETS = [
    {"corp_name": "삼성전자", "corp_code": "00126380"},
    {"corp_name": "SK하이닉스", "corp_code": "00164779"},
    {"corp_name": "LG에너지솔루션", "corp_code": "01515323"},
    {"corp_name": "삼성바이오로직스", "corp_code": "00877059"},
    {"corp_name": "현대차", "corp_code": "00164742"},
    {"corp_name": "기아", "corp_code": "00106641"},
    {"corp_name": "POSCO홀딩스", "corp_code": "00155319"},
    {"corp_name": "삼성SDI", "corp_code": "00126362"},
    {"corp_name": "LG화학", "corp_code": "00356361"},
    {"corp_name": "네이버", "corp_code": "00266961"},
]


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/3 DARTFetcher/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_json(url: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    raw = fetch_bytes(f"{url}?{query}")
    return json.loads(raw.decode("utf-8"))


def latest_business_report(corp_code: str) -> dict:
    payload = fetch_json(
        LIST_API_URL,
        {
            "crtfc_key": API_KEY,
            "corp_code": corp_code,
            "bgn_de": "20240101",
            "pblntf_ty": "A",
            "page_count": "100",
        },
    )
    if payload.get("status") != "000":
        raise RuntimeError(f"list.json failed: {payload.get('status')} {payload.get('message')}")

    reports = payload.get("list", [])
    if not reports:
        raise RuntimeError("사업보고서를 찾지 못했습니다.")

    business_reports = [
        item for item in reports if (item.get("report_nm", "").replace(" ", "")).startswith("사업보고서")
    ]
    if not business_reports:
        raise RuntimeError("사업보고서를 찾지 못했습니다.")

    business_reports.sort(
        key=lambda item: (item.get("rcept_dt", ""), item.get("rcept_no", "")),
        reverse=True,
    )
    return business_reports[0]


def extract_text_from_main_xml(zip_bytes: bytes, rcept_no: str) -> tuple[str, str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        main_name = f"{rcept_no}.xml"
        if main_name not in zf.namelist():
            candidates = [name for name in zf.namelist() if name.endswith(".xml")]
            if not candidates:
                raise RuntimeError("압축 파일에서 XML을 찾지 못했습니다.")
            main_name = sorted(candidates)[0]
        raw = zf.read(main_name)

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []
    for line in soup.get_text("\n").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return main_name, text


def save_report(corp_name: str, report: dict, xml_name: str, text: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{corp_name}.txt"
    content = "\n".join(
        [
            f"corp_name: {corp_name}",
            f"corp_code: {report.get('corp_code', '')}",
            f"stock_code: {report.get('stock_code', '')}",
            f"report_nm: {report.get('report_nm', '')}",
            f"rcept_no: {report.get('rcept_no', '')}",
            f"rcept_dt: {report.get('rcept_dt', '')}",
            f"xml_file: {xml_name}",
            f"document_url: {DOCUMENT_API_URL}?crtfc_key=***&rcept_no={report.get('rcept_no', '')}",
            "",
            text,
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    results = []
    failures = []

    for corp in CORP_TARGETS:
        corp_name = corp["corp_name"]
        corp_code = corp["corp_code"]
        try:
            report = latest_business_report(corp_code)
            rcept_no = report["rcept_no"]
            document_url = f"{DOCUMENT_API_URL}?crtfc_key={API_KEY}&rcept_no={rcept_no}"
            zip_bytes = fetch_bytes(document_url)
            xml_name, text = extract_text_from_main_xml(zip_bytes, rcept_no)
            output_path = save_report(corp_name, report, xml_name, text)

            result = {
                "corp_name": corp_name,
                "corp_code": corp_code,
                "report_nm": report.get("report_nm", ""),
                "rcept_no": rcept_no,
                "rcept_dt": report.get("rcept_dt", ""),
                "xml_file": xml_name,
                "output_path": str(output_path),
                "text_chars": len(text),
            }
            results.append(result)
            print(
                f"[OK] {corp_name}: {report.get('report_nm', '')} / "
                f"{rcept_no} / {result['text_chars']} chars"
            )
        except Exception as exc:
            failures.append({"corp_name": corp_name, "corp_code": corp_code, "error": str(exc)})
            print(f"[ERROR] {corp_name}: {exc}", file=sys.stderr)

    summary = {"results": results, "failures": failures}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print(f"saved_reports: {len(results)}")
    print(f"failed_reports: {len(failures)}")
    print(f"summary_path: {SUMMARY_PATH}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
