# Stage 0 — Dataset Provenance and Feasibility Decision

## Decision

**PIVOT for the original commercial influencer-campaign question.** None of the five candidates combines traceable real-world influencer observations with verified sales/revenue, attribution, and cost fields. The apparent sales field in the Kaggle Influencer Marketing ROI file fails the provenance gate and must not be treated as a real commercial outcome.

**GO for the narrowed Stage 1–2 research scope using the US subset of YouTube Trending Video Statistics.** The revised question is:

> Do scale-normalized interaction and timing features reveal non-random behavioral structure among US trending YouTube videos beyond raw view volume?

This is a content-performance study of a selected trending population. It is not an ROI, sales-conversion, causal campaign-effect, or general-YouTube study. “Influencer,” “campaign,” “customer,” and “purchase” claims require another dataset or a later data partnership.

Scores in [`dataset_candidate_matrix.csv`](tables/dataset_candidate_matrix.csv) use 0=does not meet, 1=partial, 2=good and are assessed against the original commercial influencer question. A high total never overrides a provenance or missing-outcome hard veto.

## Selected data: file-level verification

Source: [Kaggle dataset metadata](https://www.kaggle.com/api/v1/datasets/view/datasnaek/youtube-new), [dataset page](https://www.kaggle.com/datasets/datasnaek/youtube-new), and the linked [collection code](https://github.com/mitchelljy/Trending-YouTube-Scraper). The collector calls the YouTube Data API `videos` endpoint with `chart=mostPopular`, a region code, and `statistics,snippet`; this provides a traceable observation path for daily trending snapshots. Kaggle marks dataset version 115 as CC0; the archive was last updated on 2019-06-03. The downloader pins version 115 and verifies the exact archive hash.

The public Kaggle archive was downloaded without account credentials on 2026-11-04 and checked before extraction:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `youtube-new.zip` | 210,575,746 | `9f59324bd23ebd95aa16f854de1e0a6299bacf988b30c5e7db5f1e4216d8deda` |
| `USvideos.csv` | 62,756,152 | `09b4eb71295752705e472ebefeac9d2afab4177b7a818af795dea62744a48eb2` |
| `US_category_id.json` | 8,496 | `2e892c5a5e48d284e40fd37de0313912264041aef8833e5323bd2b2fd08c7e25` |

The actual CSV, rather than a catalog description, supplied this 16-column schema:

`video_id`, `trending_date`, `title`, `channel_title`, `category_id`, `publish_time`, `tags`, `views`, `likes`, `dislikes`, `comment_count`, `thumbnail_link`, `comments_disabled`, `ratings_disabled`, `video_error_or_removed`, `description`.

Observed file audit:

- 40,949 rows, 6,351 unique videos, 2,207 channels, and 16 categories.
- 205 trending dates from 2017-11-14 through 2018-06-14. Publication timestamps range from 2006-07-23 through 2018-06-14.
- 48 exact duplicate rows and 50 duplicate `video_id`/`trending_date` rows. Daily snapshots intentionally repeat videos, so raw rows are not independent video observations.
- 570 missing descriptions; no missing ID, date, category, or numeric metric fields.
- 633 rows with comments disabled, 169 with ratings disabled, and 23 marked error/removed. Structural disablement is not interpreted as observed zero engagement.
- No negative view/like/dislike/comment counts. Zero counts occur for 760 comment, 383 dislike, and 172 like rows.
- No sales, orders, revenue, campaign cost, attribution, customer, or ROI field exists.

The complete audit is machine-readable in [`raw_data_audit.json`](tables/raw_data_audit.json), and the raw schema is in [`raw_schema.csv`](tables/raw_schema.csv). Raw data are ignored by Git and can be reproduced with `python -m src.download_data`.

## Candidate findings

### Influencer Marketing ROI Dataset — hard veto

[Official Kaggle metadata](https://www.kaggle.com/api/v1/datasets/view/tfisthis/influencer-marketing-roi-dataset) identifies an Apache-2.0 CSV with 150,000 rows and ten fields: `campaign_id`, `platform`, `influencer_category`, `campaign_type`, `start_date`, `engagements`, `estimated_reach`, `product_sales`, `campaign_duration_days`, and `end_date`. The downloaded CSV SHA-256 was `8884cdd42684a339450ff3c926247129878b21f6f698786452fe3d6435cb46e9`.

No creator, organization, collection system, attribution method, or observational sampling process is documented. File-level patterns strongly contradict a real campaign history: IDs are exactly sequential from `CAMP100000` to `CAMP249999`; one start date is generated per row from 2022-01-01 through 2432-09-07; `product_sales` exhausts all integers 0–4,999; duration is near-uniform from 1–29; and the key numeric fields have approximately zero pairwise correlations (absolute correlations at most 0.0049). In 4.927% of rows, engagements exceed estimated reach. The page's ROI example also assumes a fixed `$500 + 0.01 × reach` cost and `$40 × sales` revenue rather than exposing measured cost/revenue fields.

The file may be useful as synthetic teaching data, but it cannot satisfy the real-life observation requirement or support real sales/ROI findings.

### SAGraph — strong audience candidate, no commercial outcome

[SAGraph paper v3](https://arxiv.org/abs/2403.15105v3) and [repository commit `4346be9`](https://github.com/xiaoqzhwhu/SAGraph/tree/4346be98c85a2ac65da0ab8977dda346197c005c) document Weibo collection across six product domains: 345,039 users, 44,182 historical influencer posts, 1,302,109 comments, and 554,629 reposts. The public Drive exposes roughly 1.08 GB across profile, interaction, and product files. The paper states CC BY 4.0 and describes anonymization, while the repository root has no license; code reuse therefore needs separate clarification.

Observed interaction/profile fields support user, influencer, and interaction analysis. They do not contain observed sales, revenue, cost, or ROI. The paper's `purchase_likelihood` is GPT-4 simulation output, not a commercial observation. Only six product campaigns exist, so campaign-level clustering would have `n=6`; a valid SAGraph study must pivot to user/influencer-level audience or network intelligence. Chinese text, privacy residuals, derived GPT-4 interests, feature-vector duplicate concentration, and an apparent feature-code issue increase review cost.

### UCI Online Retail II — real commerce, unrelated attribution

[UCI dataset 502](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (DOI `10.24432/C5CG6D`) is a real two-year UK non-store retailer transaction dataset under CC BY 4.0. The downloaded ZIP SHA-256 was `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`; its two sheets contain 1,067,371 invoice-line rows from 2009-12-01 through 2011-12-09. Quantity × price can represent transaction value after cancellation/return policies.

It contains no creator, post, campaign, reach, engagement, attribution, or campaign-cost field. It would require a full pivot to e-commerce/customer intelligence and cannot answer whether influencer engagement relates to commercial performance.

### GA4 sample ecommerce — obfuscated and inaccessible in this run

[Google's official description](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset) calls `ga4_obfuscated_sample_ecommerce` a three-month, obfuscated dataset that “emulates what a real world dataset would look like” and warns of limited internal consistency. The generic [GA4 export schema](https://developers.google.com/analytics/bigquery/event-schema) includes events, traffic source, items, and purchase revenue, but it does not prove which sample fields are populated. A table metadata request returned `401 UNAUTHENTICATED` in this environment, so the actual table was not inspected. It also lacks influencer identity and campaign cost.

## Stage 0 gate assessment

- Creator/collection path: verified for YouTube API trending snapshots.
- Actual file and schema: verified from `USvideos.csv`, with byte size and SHA-256 recorded.
- Real/synthetic/obfuscated boundary: explicitly documented for all candidates.
- Quantity and quality: measured from the selected file; repeated snapshots and structural disabled fields have explicit policies.
- Claim/field match: content-performance only; commercial claims vetoed.
- License/ethics: Kaggle labels the selected data CC0, and the table contains public video/channel metadata rather than user-profile graphs or private interactions. The catalog label does not make underlying titles/descriptions or YouTube content public domain; downstream use must also respect the [YouTube API Services Terms](https://developers.google.com/youtube/terms/api-services-terms-of-service) and applicable content rights. Trending selection and potentially removed public content remain limitations.

The Stage 0 status is therefore **PIVOT**, followed by a scoped **GO** to Stage 1–2 on the verified YouTube content-performance question. Returning to the original commercial question requires a separately sourced, permissioned dataset with measured attribution and commercial outcomes.
