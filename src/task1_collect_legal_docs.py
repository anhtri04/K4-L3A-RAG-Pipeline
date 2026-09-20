"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Chủ đề: Quản trị và chính sách bóng đá (Football Governance & Regulations).
Nguồn: UK Department for Culture, Media and Sport (DCMS) công khai theo OGL v3.0.
"""

from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_SOURCES = {
    "football_governance_significant_influence_control_guidance_2025.pdf": (
        "https://assets.publishing.service.gov.uk/media/693315275b5198836f30415a/"
        "E03461288_Statutory_Guidance_About_the_Meaning_of__Significant_Influence_or_Control__final_accessible_December_2025__1_.pdf"
    ),
    "fan_led_review_football_governance_2021.pdf": (
        "https://assets.publishing.service.gov.uk/media/63e4d010d3bf7f05b871200d/"
        "Football_Fan_led_Governance_Review_v8Web_Accessible.pdf"
    ),
    "club_football_governance_white_paper_2023.pdf": (
        "https://assets.publishing.service.gov.uk/media/63f65d3de90e077bb0c92853/"
        "Reform_of_club_football_governance_-_White_Paper.pdf"
    ),
    "government_response_womens_football_2023.pdf": (
        "https://assets.publishing.service.gov.uk/media/656a1e710f12ef070e3e0104/"
        "Government_response_to_independent_review_-_reframing_the_opportunity_in_women_s_football.pdf"
    ),
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải và kiểm tra ít nhất 3 PDF từ nguồn công khai."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
    }
    for filename, url in LEGAL_SOURCES.items():
        target_path = DATA_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 1024:
            print(f"Already verified: {filename} ({target_path.stat().st_size:,} bytes)")
            continue

        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            target_path.write_bytes(response.content)
            print(f"Downloaded: {filename} ({len(response.content):,} bytes)")
        except Exception as error:
            print(f"Download failed for {filename}: {error}")


if __name__ == "__main__":
    setup_directory()
    download_documents()
