import platform
import re
import time
import urllib.parse
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from kiwipiepy import Kiwi
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from transformers import pipeline


# ----------------------------------------------------
# 1. 폰트 설정 및 모델/드라이버 초기화
# ----------------------------------------------------
def set_korean_font():
    """OS 환경에 맞춰 Matplotlib 한글 폰트 및 마이너스 부호 설정"""
    os_name = platform.system()
    if os_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif os_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource
def load_sentiment_model():
    """공개 한국어 감정분석 사전학습 모델 로드"""
    model_name = "matthewburke/korean_sentiment"
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=128,
    )


@st.cache_resource
def load_kiwi():
    """단어 추출을 위한 Kiwipiepy 형태소 분석기 로드"""
    return Kiwi()


def init_webdriver():
    """Selenium 크롬 드라이버 초기화 (자동화 탐지 우회 옵션 적용)"""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation"]
    )
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        },
    )
    return driver


# ----------------------------------------------------
# 2. 데이터 수집 함수 (유튜브, 네이버 영화, CGV)
# ----------------------------------------------------
def extract_video_id(url: str) -> str:
    """유튜브 URL에서 비디오 ID 추출"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def get_video_comments(
    api_key: str, video_id: str, max_results: int = 50
) -> list:
    """YouTube Data API v3를 호출하여 댓글 수집 및 익명화"""
    youtube = build("youtube", "v3", developerKey=api_key)
    response = (
        youtube.commentThreads()
        .list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            order="relevance",
        )
        .execute()
    )
    comments = []
    for idx, item in enumerate(response.get("items", []), start=1):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append(
            {
                "작성자": f"user_{idx:03d}",
                "댓글 내용": snippet.get("textDisplay", ""),
                "좋아요 수": snippet.get("likeCount", 0),
                "작성일": snippet.get("publishedAt", ""),
            }
        )
    return comments


def get_naver_movie_reviews(
    movie_title_or_url: str, max_results: int = 50
) -> list:
    """Selenium을 사용하여 네이버 영화 관람평 수집 (기존 코드 유지)"""
    driver = init_webdriver()
    reviews = []
    seen_texts = set()

    try:
        if movie_title_or_url.startswith("http"):
            url = movie_title_or_url
        else:
            query = urllib.parse.quote(f"{movie_title_or_url} 평점")
            url = f"https://search.naver.com/search.naver?query={query}"

        driver.get(url)

        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "span[class*='_commentFlow']")
            )
        )

        page_num = 1
        max_pages = (max_results // 5) + 15

        while len(reviews) < max_results and page_num <= max_pages:
            comment_elements = driver.find_elements(
                By.CSS_SELECTOR, "span[class*='_commentFlow']"
            )

            for comment_el in comment_elements:
                if len(reviews) >= max_results:
                    break

                try:
                    try:
                        driver.execute_script(
                            "arguments[0].click();", comment_el
                        )
                    except Exception:
                        pass

                    text = comment_el.text.strip()
                    if not text or text in seen_texts:
                        continue

                    seen_texts.add(text)
                    idx = len(reviews) + 1
                    reviews.append(
                        {
                            "작성자": f"naver_user_{idx:03d}",
                            "댓글 내용": text,
                            "좋아요 수": 0,
                            "작성일": "-",
                        }
                    )
                except Exception:
                    continue

            if len(reviews) >= max_results:
                break

            try:
                move_btns = driver.find_elements(
                    By.CSS_SELECTOR,
                    "div[class*='_pagination'] button[class*='_moveBtn']",
                )

                if len(move_btns) >= 2:
                    next_btn = move_btns[1]
                elif len(move_btns) == 1:
                    next_btn = move_btns[0]
                else:
                    break

                if (
                    next_btn.get_attribute("disabled") is not None
                    or next_btn.get_attribute("aria-disabled") == "true"
                ):
                    break

                first_review_before = (
                    comment_elements[0].text if comment_elements else ""
                )

                driver.execute_script("arguments[0].click();", next_btn)
                page_num += 1

                for _ in range(15):
                    time.sleep(0.2)
                    new_comments = driver.find_elements(
                        By.CSS_SELECTOR, "span[class*='_commentFlow']"
                    )
                    if (
                        new_comments
                        and new_comments[0].text != first_review_before
                    ):
                        break
            except Exception:
                break

    finally:
        driver.quit()

    return reviews


def get_cgv_movie_reviews(url: str, max_results: int = 50) -> list:
    """Selenium을 사용하여 CGV 실관람평 영역을 반복 스크롤하며 수집"""
    driver = init_webdriver()
    reviews = []
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(4.0)

        # '실관람평' 탭 클릭
        try:
            review_tab = driver.find_element(
                By.XPATH, "//*[contains(text(), '실관람평')]"
            )
            driver.execute_script("arguments[0].click();", review_tab)
            time.sleep(3.0)
        except Exception:
            pass

        # 목표 개수(50개) 이상 로딩될 때까지 반복 스크롤
        body = driver.find_element(By.TAG_NAME, "body")
        prev_count = 0
        same_count_streak = 0

        for _ in range(35):
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            body.send_keys(Keys.END)
            driver.execute_script(
                """
                const divs = document.querySelectorAll('div');
                for (let d of divs) {
                    if (d.scrollHeight > d.clientHeight + 100) {
                        d.scrollTop = d.scrollHeight;
                    }
                }
                """
            )
            time.sleep(1.5)

            loaded_cards = driver.find_elements(
                By.XPATH, "//*[contains(text(), '도움됐어요')]"
            )
            curr_count = len(loaded_cards)
            if curr_count >= max_results:
                break

            if curr_count == prev_count:
                same_count_streak += 1
                if same_count_streak >= 5:
                    break
            else:
                same_count_streak = 0
            prev_count = curr_count

        # '도움됐어요' 버튼이 포함된 각 리뷰 카드 블록 추출
        review_cards = driver.find_elements(
            By.XPATH,
            "//button[contains(., '도움됐어요')]/ancestor::div[1]/parent::div",
        )
        if not review_cards:
            review_cards = driver.find_elements(
                By.XPATH,
                "//*[contains(text(), '도움됐어요')]/ancestor::li | //*[contains(text(), '도움됐어요')]/ancestor::div[2]",
            )

        extracted_texts = []
        for card in review_cards:
            lines = [
                line.strip() for line in card.text.split("\n") if line.strip()
            ]
            filtered_lines = []
            for idx_line, line in enumerate(lines):
                if idx_line == 0:
                    continue
                if (
                    re.search(
                        r"(전|일|시간|분|방금|IMAX|4DX|SCREENX|2D|3D|일반)", line
                    )
                    and len(line) < 25
                ):
                    continue
                if (
                    line in ["좋았어요", "아쉬웠어요"]
                    or "도움됐어요" in line
                    or line == "⋮"
                ):
                    continue
                filtered_lines.append(line)

            if filtered_lines:
                review_text = " ".join(filtered_lines)
                if review_text not in extracted_texts:
                    extracted_texts.append(review_text)

        for idx, text in enumerate(extracted_texts[:max_results], start=1):
            reviews.append(
                {
                    "작성자": f"cgv_user_{idx:03d}",
                    "댓글 내용": text,
                    "좋아요 수": 0,
                    "작성일": "-",
                }
            )
    finally:
        driver.quit()
    return reviews


# ----------------------------------------------------
# 3. 텍스트 전처리, AI 모델 감정 분석 및 단어 추출 함수
# ----------------------------------------------------
STOPWORDS = {
    # 명사 및 부사 불용어
    "영화",
    "호프",
    "작품",
    "사람",
    "생각",
    "느낌",
    "부분",
    "정도",
    "진짜",
    "정말",
    "너무",
    "완전",
    "조금",
    "약간",
    "그냥",
    # 동사·형용사 어간 불용어 (Kiwi 토큰화 결과 기준 '-다' 제외)
    "하",
    "되",
    "있",
    "없",
    "보",
    "같",
    "않",
    "그렇",
    "이렇",
    "저렇",
    "오",
    "가",
    "주",
    "받",
    "만들",
    "나오",
    "들",
}


def clean_comment_text(text: str) -> str:
    """1~2단계 전처리: URL, 타임스탬프, HTML, 이모지 제거 및 반복 문자 정규화"""
    if not isinstance(text, str):
        return ""

    # HTML 태그 및 엔티티 제거
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)

    # URL 제거
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 유튜브 타임스탬프 제거 (예: 01:23, 1:02:33)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)

    # 한글, 영문, 숫자, 기본 문장부호(.,!?~) 외 이모지 및 특수기호 제거
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,!?~]", " ", text)

    # 3회 이상 반복되는 문자 정규화 (예: ㅋㅋㅋㅋㅋ -> ㅋㅋ, ㅠㅠㅠㅠ -> ㅠㅠ, !!!!! -> !!)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 연속 공백 및 줄바꿈을 단일 공백으로 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """3단계 전처리: 텍스트 정제 후 무의미한 단답(3자 미만) 및 중복 댓글 필터링"""
    df = df.copy()
    df["전처리 댓글"] = df["댓글 내용"].apply(clean_comment_text)

    # 정제 후 길이가 3자 미만인 무의미 댓글 제외
    df = df[df["전처리 댓글"].str.len() >= 3]

    # 전처리된 텍스트 기준 중복 댓글 제거
    df = df.drop_duplicates(subset=["전처리 댓글"]).reset_index(drop=True)
    return df


def extract_keywords_from_comment(text: str, kiwi: Kiwi) -> list:
    """4단계 전처리: 불용어(Stopwords)를 제외하고 의미 있는 명사, 형용사, 동사 어간 추출"""
    tokens = kiwi.tokenize(text)
    valid_tags = {"NNG", "NNP", "VA", "VV", "XR"}
    keywords = []
    for token in tokens:
        if token.tag not in valid_tags or token.form in STOPWORDS:
            continue

        # 동사(VV)·형용사(VA)는 1글자 감정 어간('좋', '싫' 등)을 살리고, 명사·어근은 2글자 이상만 추출
        is_valid_length = (
            len(token.form) >= 1
            if token.tag in {"VA", "VV"}
            else len(token.form) > 1
        )

        if is_valid_length and token.form not in keywords:
            keywords.append(token.form)
    return keywords


def analyze_dataframe_sentiment(
    df: pd.DataFrame, classifier, kiwi: Kiwi
) -> pd.DataFrame:
    """전처리된 텍스트를 바탕으로 AI 감정 판별, 점수 계산 및 키워드 추출"""
    df = preprocess_dataframe(df)
    if df.empty:
        return df

    sentiments = []
    scores = []
    extracted_words_list = []

    for clean_text in df["전처리 댓글"]:
        result = classifier(clean_text)[0]
        label_raw = result["label"].upper()
        confidence = result["score"]

        if label_raw in ["LABEL_1", "POSITIVE", "1"]:
            label = "긍정"
            sentiment_score = round(confidence * 100, 2)
        else:
            label = "부정"
            sentiment_score = round(-confidence * 100, 2)

        keywords = extract_keywords_from_comment(clean_text, kiwi)

        sentiments.append(label)
        scores.append(sentiment_score)
        extracted_words_list.append(", ".join(keywords) if keywords else "-")

    df["감정"] = sentiments
    df["감성 점수"] = scores
    df["추출 단어"] = extracted_words_list
    return df


# ----------------------------------------------------
# 4. 시각화 함수
# ----------------------------------------------------
def draw_sentiment_pie(df: pd.DataFrame, title: str):
    """플랫폼/영상별 감정 비율 원형 그래프 시각화"""
    counts = df["감정"].value_counts()
    pos_count = counts.get("긍정", 0)
    neg_count = counts.get("부정", 0)

    if pos_count == 0 and neg_count == 0:
        st.write("표시할 감정 데이터가 없습니다.")
        return

    labels = ["긍정", "부정"]
    sizes = [pos_count, neg_count]
    colors = ["#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------
# 5. 메인 UI
# ----------------------------------------------------
def main():
    set_korean_font()
    st.title("영화 호프로 알아보는 심리학")

    classifier = load_sentiment_model()
    kiwi = load_kiwi()

    # 유튜브 수집 설정
    st.subheader("1. 유튜브 댓글 수집 설정")
    api_key = st.text_input(
        "YouTube Data API v3 키를 입력하세요 (유튜브 분석 시 필수)",
        type="password",
    )
    url_1 = st.text_input(
        "유튜브 중립 URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_1",
    )
    url_2 = st.text_input(
        "유튜브 부정 URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_2",
    )
    url_3 = st.text_input(
        "유튜브 긍정 URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="url_3",
    )

    # 웹 크롤링 설정 (네이버 영화 / CGV)
    st.subheader("2. 영화 리뷰 웹사이트 수집 설정")
    naver_target = st.text_input(
        "네이버 영화 제목 (또는 리뷰 URL)",
        placeholder="예: 호프 (또는 네이버 평점 검색 URL)",
        key="naver_target",
    )
    cgv_url = st.text_input(
        "CGV 영화 리뷰 URL",
        placeholder="https://cgv.co.kr/cnm/cgvChart/movieChart/30001297",
        key="cgv_url",
    )

    if st.button("분석 시작"):
        youtube_inputs = [
            ("유튜브 중립", url_1),
            ("유튜브 부정", url_2),
            ("유튜브 긍정", url_3),
        ]
        has_youtube = any(u.strip() for _, u in youtube_inputs)
        has_naver = bool(naver_target.strip())
        has_cgv = bool(cgv_url.strip())

        if not (has_youtube or has_naver or has_cgv):
            st.error(
                "유튜브, 네이버 영화, CGV 중 최소 하나의 대상을 입력해야 합니다."
            )
            return

        if has_youtube and not api_key:
            st.error("유튜브 URL을 분석하려면 API 키를 입력해야 합니다.")
            return

        tasks = []
        for yt_label, u in youtube_inputs:
            if u.strip():
                vid = extract_video_id(u.strip())
                if vid:
                    tasks.append(
                        {"type": "youtube", "label": yt_label, "target": vid}
                    )
                else:
                    st.warning(f"{yt_label}의 비디오 ID를 확인할 수 없습니다.")

        if has_naver:
            tasks.append(
                {
                    "type": "naver",
                    "label": "네이버 영화",
                    "target": naver_target.strip(),
                }
            )

        if has_cgv:
            tasks.append(
                {"type": "cgv", "label": "CGV 영화", "target": cgv_url.strip()}
            )

        if not tasks:
            st.error("수집 가능한 유효한 대상이 없습니다.")
            return

        progress_bar = st.progress(0.0)
        collected_data = {}
        total_accumulated = 0

        with st.status("데이터 수집 및 AI 분석 진행 중...", expanded=True) as status:
            for i, task in enumerate(tasks):
                label = task["label"]
                target = task["target"]
                t_type = task["type"]
                status.write(f"[{label}] 데이터 수집, 전처리 및 감정 분석 시작")

                try:
                    if t_type == "youtube":
                        comments = get_video_comments(
                            api_key, target, max_results=50
                        )
                    elif t_type == "naver":
                        comments = get_naver_movie_reviews(
                            target, max_results=50
                        )
                    elif t_type == "cgv":
                        comments = get_cgv_movie_reviews(target, max_results=50)
                    else:
                        comments = []

                    raw_count = len(comments)

                    if raw_count > 0:
                        raw_df = pd.DataFrame(comments)
                        analyzed_df = analyze_dataframe_sentiment(
                            raw_df, classifier, kiwi
                        )
                        valid_count = len(analyzed_df)
                        total_accumulated += valid_count

                        if valid_count > 0:
                            collected_data[label] = analyzed_df

                        status.write(
                            f"[{label}] 완료: 수집 {raw_count}건 → 전처리 후 유효 {valid_count}건 | 누적 {total_accumulated}건"
                        )
                    else:
                        status.write(f"[{label}] 완료: 수집 0건")

                except HttpError as e:
                    status.write(
                        f"[{label}] 유튜브 API 오류 (상태 코드: {e.resp.status})"
                    )
                except Exception as e:
                    status.write(f"[{label}] 수집 오류 발생 (원인: {str(e)})")

                progress_bar.progress((i + 1) / len(tasks))

            status.update(
                label=f"작업 완료 (전처리 후 총 {total_accumulated}건 분석 완료)",
                state="complete",
                expanded=False,
            )

        if collected_data:
            tab1, tab2 = st.tabs(["감정 분석 결과", "수집 데이터 원본"])

            with tab1:
                st.subheader("1. 본문 수집 단어 및 감정 판별 (상위 20개 표기)")
                for label, df in collected_data.items():
                    with st.expander(
                        f"[{label}] (전처리 후 {len(df)}건 중 상위 20건 표시)",
                        expanded=False,
                    ):
                        st.dataframe(
                            df[
                                [
                                    "전처리 댓글",
                                    "추출 단어",
                                    "감정",
                                    "감성 점수",
                                ]
                            ].head(20),
                            use_container_width=True,
                        )

                st.subheader("2. 감성 점수 계산 요약 (전처리 완료 데이터 기준)")
                summary_list = []
                for label, df in collected_data.items():
                    summary_list.append(
                        {
                            "출처": label,
                            "유효 리뷰 수": len(df),
                            "평균 감성 점수": round(df["감성 점수"].mean(), 2),
                            "긍정 리뷰 수": (df["감정"] == "긍정").sum(),
                            "부정 리뷰 수": (df["감정"] == "부정").sum(),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(summary_list), use_container_width=True
                )

                st.subheader("3. 출처별 감정 분석 원형 그래프")
                youtube_items = [
                    (label, df)
                    for label, df in collected_data.items()
                    if label.startswith("유튜브")
                ]
                movie_site_items = [
                    (label, df)
                    for label, df in collected_data.items()
                    if not label.startswith("유튜브")
                ]

                if youtube_items:
                    yt_cols = st.columns(len(youtube_items))
                    for col, (label, df) in zip(yt_cols, youtube_items):
                        with col:
                            draw_sentiment_pie(df, f"{label} 감정 비율")

                if movie_site_items:
                    site_cols = st.columns(len(movie_site_items))
                    for col, (label, df) in zip(site_cols, movie_site_items):
                        with col:
                            draw_sentiment_pie(df, f"{label} 감정 비율")

            with tab2:
                st.subheader("수집된 원본 및 전처리 비교 데이터 (상위 20개 표기)")
                for label, df in collected_data.items():
                    st.write(f"[{label}] 데이터 (상위 20건)")
                    st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()