# Beyond Engagement — Codex Research & Implementation Handoff

> **Proje tipi:** Yüksek lisans / Data Mining final projesi  
> **Çalışma adı:** *Beyond Engagement: A Multi-Method Data Mining Framework for Influencer Marketing Intelligence*  
> **Belge durumu:** Araştırma planı; hipotezler ve eşikler önceden önerilmiştir. Dataset ve deney sonuçları **henüz doğrulanmamıştır**.  
> **Öncelikli çalışma kapsamı:** Stage 0 (veri seçimi / gerçeklik doğrulaması), Stage 1 / H1 (EDA ve feature engineering), Stage 2 / H2 (clustering tendency).  
> **Dil:** Açıklamalar Türkçe olabilir; kod, değişkenler, grafik başlıkları ve paper taslağı İngilizce olsun.

## 1. Bağlam ve dersin beklentileri

Bu çalışma, CSE454/BIL557 **Data Mining** yüksek lisans dersi için hazırlanıyor. Derste veri ön işleme, association/frequent pattern mining, classification, clustering, dimensionality reduction, anomaly detection, graph mining, text mining ve data warehouses işleniyor. Chapter 8'de K-Means, K-Medoids/PAM/CLARA/CLARANS, hierarchical clustering, BIRCH/CF Tree, DBSCAN, grid-based methods ve clustering evaluation bulunuyor.

Final proje için istenenler: **gerçek yaşam verisi**, birden fazla data-mining yöntemi **uygulaması ve karşılaştırması**, akademik paper, örnekler/grafikler, sunum ve çalışır demo/uygulama. Ders kapsamında işlenmeyen yöntemler kullanılabilir. Syllabus'ta başka derslerin projelerini aynen sunmama koşulu vardır. Gerçek veri, tekrar üretilebilir deney ve dürüst raporlama her şeyden önemlidir.

Orijinal planda 4–14. haftalar arası takvim vardı. **Mevcut haftayı ve kalan süreyi repository/öğrencinin güncel bilgisine göre güncelle; eski hafta numaralarını bugünün durumu sanma.**

## 2. Asıl iş problemi ve ürün vizyonu

Markalar influencerları çoğunlukla follower count, reach ve engagement gibi yüzeysel göstergelerle değerlendiriyor. Bunlar her zaman ticari etkiyi açıklamıyor. Örneğin çok etkileşim alan kampanya az satış üretebilir; daha niş ve küçük kampanya, bütçe başına daha güçlü sonuç verebilir.

**Ana araştırma sorusu (RQ-Main):**
> *Can multi-method data mining reveal commercially meaningful influencer campaign patterns that are not captured by conventional reach/engagement-based evaluation?*

**Proje çıktısı:** Kampanya davranış arketipleri, belirli koşullarla birlikte görülen sonuçlar, sıra dışı performans örnekleri, yöntem karşılaştırmaları ve (veri uygunsa) tahmin modeli. Bunlar markalara karar desteği sağlayabilecek araştırma bulgularıdır; **otomatik olarak kanıtlanmış ROI veya nedensel etki olarak sunulamaz**.

**Uzun vadeli ürün vizyonu:** Influencer'ın Instagram/YouTube, yorum, community chat, canlı yayın, izinli first-party site etkinlikleri, UTM/promo attribution ve satış verisini birleştirerek niş kitle içgörüleri oluşturması. **Bu dönem implementasyonunda cross-site cookie tracking, Instagram scraping, kullanıcı profilleme altyapısı ve ürünleştirme yoktur.** Gizlilik ve erişim kurallarına uygun, izinli/anonim toplulaştırılmış veriler tercih edilir.

**En kritik ayrım:**
- **Campaign intelligence:** Kampanyalara ait performans alanlarından pazarlama örüntüleri çıkarma. Satış/ROI ancak ilgili gerçek alanlar varsa ölçülebilir.
- **Audience intelligence:** Yorumlar, kullanıcı profilleri veya etkileşim ağlarından ilgi alanı/niche keşfi. Kampanya seviyesindeki basit tablolar bunun için yeterli değildir.

Dataset ikisini de desteklemiyorsa iddiaları genişletme. Özellikle **satış verisi bulunmayan veriyle sales conversion veya ROI sonucu iddia etme**.

## 3. Araştırma yapısı ve hipotez ağacı

| Stage | Hipotez / soru | Ana çıktı |
|---|---|---|
| 0 / H0 | Veri gerçekten gözlemlenmiş, kaynağı izlenebilir ve araştırma sorusunu taşıyabiliyor mu? | Dataset audit ve GO/PIVOT kararı |
| 1 / H1 | Ham metriklerden türetilen, iş anlamı taşıyan özellikler tutarlı ve bilgi verici mi? | Data dictionary, EDA, feature table |
| 2 / H2 | Analiz uzayında rastgele dağılımdan farklı, anlamlı bir gruplaşma eğilimi var mı? | Hopkins + null-distance + görsel inceleme |
| 3 / H3 | Yorumlanabilir, kararlı kampanya arketipleri çıkarılabiliyor mu? Hangi clustering daha iyi? | K-Means / K-Medoids / BIRCH / DBSCAN karşılaştırması |
| 4 / H4 | Varyans az sayıda bileşenle temsil edilebiliyor mu? | PCA explained variance / grafikler |
| 5 / H5 | Bazı platform–niche–kampanya kombinasyonları güçlü sonuçlarla birlikte görülüyor mu? | Association rules, support/confidence/lift ve holdout test |
| 6 / H6 | Engagement ile ticari performans arasında sıra dışı uyuşmazlıklar var mı? | Isolation Forest / LOF karşılaştırması |
| 7 / H7a | Kampanya başarısı uygun pre-campaign değişkenlerle basit baseline'dan iyi tahmin edilebilir mi? | Classification baseline comparison |
| 8 / H7b-H8 | Mining-derived features, klasik reach/engagement sinyallerinin ötesinde ek fayda sağlıyor mu? | Ablation, incremental value, final assessment |

**Önemli:** H0, H1, H2 şu anda *araştırılacak* hipotezlerdir; henüz doğrulanmış bulgular değildir. Sonraki stage'lere otomatik geçme.

## 4. Stage 0 — Gerçek dataset seçimi ve kabul kararı [ŞİMDİ]

### 4.1. Dataset adayları (kaynaklar doğrulanmalı)

| Aday | Kaynak | Olası avantaj | Kritik risk / eksik |
|---|---|---|---|
| Influencer Marketing ROI Dataset | https://www.kaggle.com/datasets/tfisthis/influencer-marketing-roi-dataset | Platform, kampanya, engagement ve muhtemel satış/outcome alanları | **Gözlemlenmiş gerçek kampanya verisi olup olmadığı henüz doğrulanmadı.** Kaynak açıklamasını ve dosyayı incele; sentetikse 'real-life observations' diye kullanma. |
| SAGraph (Weibo influencer advertising) | https://github.com/xiaoqzhwhu/SAGraph/ ; https://arxiv.org/abs/2403.15105 | Gerçek sosyal ağ/yorum/paylaşım ve profil bağlamı; özgün 'niche audience' vizyonuna yakın | Satış/ROI görünmüyor; README ve makale sayıları farklı sürümleri yansıtabilir; indirme hakkı, erişim, dil ve dosya büyüklüğü doğrulanmalı. Kullanıcı profillerindeki alanların türetilmiş/simüle olup olmadığı kontrol edilmeli. |
| YouTube Trending Video Statistics | https://www.kaggle.com/datasets/datasnaek/youtube-new | YouTube API'den toplanmış video performansı; erişilebilir çalışma | Yalnızca trending selection bias; doğrudan satış/ROI ve tam takipçi profili yok. |
| UCI Online Retail II | https://archive.ics.uci.edu/dataset/502/online+retail+ii | Doğrulanmış gerçek transaction verisi, 1,067,371 kayıt, açık akademik kaynak | Influencer verisi DEĞİL; seçilirse araştırma başlığı dürüstçe e-commerce pattern/customer intelligence'a dönmelidir. |
| GA4 sample ecommerce | https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset | Gerçek e-ticaret event yapısı / funnel analizine uygun | Google tarafından **obfuscated/emulated** olarak sunuluyor, yer yer tutarsız; strict real-life gözlem ölçütü için dikkatli değerlendirilmeli. |

**Not:** Bu tablo ilk taramadır. URL/erişim, gerçeklik, lisans, zaman aralığı, satır sayısı, şema ve veri kalitesi **Codex tarafından kontrol edilecektir**. Kişisel hesap veya API anahtarı gerektiren kaynağa izinsiz girişme. Repo için bütçe/indirme kapasitesini gözet.

### 4.2. Karar matrisi

Dataset başına aşağıdakileri raporla (0=karşılamıyor, 1=kısmen, 2=iyi):

1. **Authenticity / provenance:** gerçek gözlemlenmiş veri mi, yaratıcı ve toplama yöntemi izlenebilir mi?
2. **Relevance:** influencer campaign veya audience research sorusuna uygun mu?
3. **Outcome availability:** sales/ROI gerçek ölçüm mü, yoksa yalnızca views/likes mı?
4. **Granularity:** campaign/post/user/interaction düzeyi ayrılabiliyor mu?
5. **Diversity:** kategoriler, platformlar, farklı dönemler ve davranış varyansı yeterli mi?
6. **Access and ethics:** indirme izni, lisans, kişisel veri riski, anonimleştirme durumu?
7. **Feasibility:** zaman, hesaplama, dil ve temizleme iş yükü uygun mu?

**Hard veto:** kaynak gerçekliği belirsizse veya istenen outcome dataset'te yoksa, ilgili iddia için GO yok. Bir seçeneğin toplam puanı yüksek diye hard veto aşılmaz.

### 4.3. G0 success criteria / gate

- [ ] Veri yaratıcısı/toplama yolu ve gözlem türü belgelenmiş.
- [ ] Dataset dosyası veya güvenilir erişim yolu gerçekten denetlenmiş; kolonlar **dosya içinden** doğrulanmış.
- [ ] Gerçek/sentetik/obfuscated ayrımı açıkça yazılmış.
- [ ] Kullanılabilir veri miktarı, eksikler, tekrarlar ve sınıf/alan çeşitliliği ölçülmüş.
- [ ] Hedeflenen iddialar ile dataset alanları eşleşiyor (özellikle satış iddiası).
- [ ] Lisans, attribution, etik ve privacy değerlendirilmiş.
- [ ] Adaylar `reports/dataset_decision.md` içinde gerekçeli karşılaştırılmış.

**GO:** doğrulanmış, uygun ve erişilebilir dataset seçildi.  
**REVIEW:** veri var ama outcome/provenance/kalite belirsiz; karar için gereken testi tamamla.  
**PIVOT:** yalnızca sentetik veya ana soruya uygun olmayan kaynaklar kaldı; araştırma sorusunu ve başlığı şeffaf biçimde yeniden daralt, orijinal amaç değişiyorsa kullanıcıya açıkça bildir.

## 5. Stage 1 / H1 — EDA ve feature engineering [ŞİMDİ]

**Hipotez H1 (daha sınanabilir biçim):**
> *Domain-informed, scale-normalized features capture meaningful behavioral heterogeneity beyond raw volume measures.*

**Önce analiz birimini belirle:** campaign, creator, post/video, user veya session. Farklı grained tabloları yanlış join'leyerek satır çoğaltma. Mümkünse campaign-level çalış; başka dataset seçildiyse açıkça yeniden tanımla.

**Yapılacaklar:**
- Ham dosyanın değişmez kopyası; kaynak URL ve SHA-256 kaydı, şema, data dictionary.
- Satır/kolon sayısı, veri türleri, missing values, duplicates, basic distributions, zero denominators, date coverage, category cardinality.
- Her değişkeni `ID / predictor / outcome / derived outcome / leakage-risk / metadata` diye sınıflandır.
- Negatif satış, iade, bot olasılığı, duplicated post events, yanlış tarih, duplicate videos gibi dataset'e özgü kalite sorunlarını kayıt altına al.
- İş anlamı olan türev değişkenleri, **yalnızca alanlar gerçekten mevcutsa**, üret:
  - `engagement_per_reach = engagements / reach` (sıfır payda kontrolü)
  - `sales_per_reach = sales / reach` (**conversion rate diye adlandırma**)
  - `sales_per_engagement = sales / engagements` (**purchase conversion diye adlandırma**)
  - `cost_per_sale = campaign_cost / sales` (maliyet varsa)
  - `engagement_per_day = engagements / duration_days` (pozitif süre varsa)
  - `log1p_reach`, `log1p_engagements` vb. çarpık dağılımlar için
- Eğer video/comment/graph dataset seçildiyse uygun davranışsal özelliklere yönel; satış alanını hayal etme.
- Oranların dağılımını, düşük payda riskini, doğrudan ham sütunlarla redundant olup olmadıklarını, kategori ve zaman dilimi farklılıklarını incele.
- Ham volume features ile normalize/engineered features set'lerini karşılaştır. İlk aşamada histograms, boxplots, correlation heatmap, zero/missingness ve segment karşılaştırmaları yeterli.
- *Scale* gerekiyorsa yalnızca train partisyonunda fit edilen scaler kullan; outlier için robust scaler/log transform değerlendirmesi yap.

**H1 değerlendirme planı:**
- H1'in 'daha iyi bilgi' iddiasını sırf bir histogram veya korelasyonla kanıtlanmış sayma.
- H1 için ön kanıt: temiz, varyasyon içeren, yorumlanabilir yeni metrikler; ham ölçütlerden farklı sıralamalar/profiller ve örnek karşılaştırmalar.
- Formal karşılaştırma Stage 3 ve Stage 8'de raw-only vs engineered features şeklinde yapılacak. Bu yüzden H1 gate'inin sonucu **engineering readiness**, nihai bilimsel kabul değil.

**G1 success criteria:**
- [ ] Veri sözlüğü ve granularity net.
- [ ] En az 4 iş açısından anlamlı ve geçerli türev özellik **veri elveriyorsa** oluşturuldu; uymuyorsa sayı zorlanmadı, gerekçe yazıldı.
- [ ] Sıfır payda, çarpık dağılım, missing ve negatif/uygunsuz değer politikaları açık.
- [ ] En az 5 yorumlanabilir EDA grafiği ve açıklamalı özet tablo üretildi.
- [ ] Leakage / circular metric risk matrisi yazıldı; feature ve outcome ayrıldı.
- [ ] Temiz analiz tablosu ve tekrar üretilebilir script hazır.
- [ ] Raw-vs-engineered karşılaştırması için gelecek deney tasarımı belirlendi.

**GO:** geçerli analiz tablosu + güçlü data dictionary + denetlenebilir feature tanımları.  
**REVIEW:** tekrar eden alanlar, gizli label leakage, bozuk payda, doğrulanmamış join.  
**STOP/PIVOT:** araştırma sorusunu destekleyecek ölçüm yok.

## 6. Stage 2 / H2 — Clustering tendency [ŞİMDİ]

**Hipotez H2:**
> *The chosen campaign/creator behavioral feature space exhibits non-random grouping structure under a justified null model.*

**Kritik tasarım uyarısı:** Eğer clusterları `sales_per_reach` ve `engagement_per_reach` ile oluşturursak, sonradan "high sales vs low engagement" diye bu clusterları yorumlamak **kısmen tanım gereği** gerçekleşir. Bunu bağımsız doğrulama veya yaratıcı keşif diye gösterme. İki ayrı deney tasarımı kaydet:

- **Descriptive clustering:** outcome dahil geçmiş kampanya özellikleriyle arketipleri keşfeder. Yorumlanabilirlik sağlar, prediction veya dış doğrulama iddiası yok.
- **Predictor-only clustering:** pre-campaign/uygun predictor özelliklerinden kümeler oluşturur, satış/outcome verisini *sonradan dış ölçüt* olarak kıyaslar. Gerçek incremental marketing value değerlendirmesi için tercih edilir.

**Clustering tendency ölçüm planı:**
1. `X_numeric` için uygun imputation, categorical encoding, scaling ve gerekiyorsa near-duplicate filtreleri.
2. Random-seed sabit; birden fazla temsilî alt örneklemle **Hopkins Statistic** hesapla; kullanılan formülün oryantasyonunu kontrol et (`~0.5`: random; `~1`: clustering tendency yaygın konvansiyon).
3. İkinci bağımsız kanıt olarak standardized nearest-neighbor distance dağılımını, veriye uygun random/null örnekleriyle kıyasla. **Null dağılımın seçimi ve varsayımları açıkça yazılsın.**
4. PCA 2D yalnızca keşif/görselleştirme içindir, istatistiksel kanıt olarak yeterli değildir. Açıklanan varyansı raporla.
5. Feature scaling, düşük/çok yüksek dimensionality, categorical encoding ve duplicate concentration etkilerine karşı sensitivity analysis yap.
6. Hopkins sonuçlarını farklı random seed ve sample büyüklüklerinde raporla (mean, std ve örnekler).
7. Veri temsilinin clusterability için uygun olup olmadığını sorgula; gerekirse feature set'i iyileştir, ama sonucu zorlayacak manipülasyon yapma.

**Önceden belirlenen operasyonel sinyaller (evrensel cutoff değil):**
- `Hopkins >= 0.65` çoğu tekrarda: araştırmaya değer sinyal; **tek başına GO değildir**.
- `Hopkins ~ 0.5`: rastgele dağılımla uyumlu olabilir; null bağımsız kanıta bak.
- Tek bir PCA dağılımı veya tek Silhouette skoru "natural clusters exist" ispatlamaz.

**G2 success criteria:**
- [ ] Reproducible Hopkins implementation veya güvenilir paket + hesaplama konvansiyonu test edilmiş.
- [ ] Birden fazla seed/sample denemesi, ortalama ve değişkenlik raporlanmış.
- [ ] Null comparison / distance-distribution analizi yapılmış.
- [ ] Feature set ve dönüşümlerin etkisi raporlanmış.
- [ ] Sonuç `GO / REVIEW / PIVOT` ve **destekleyen kanıtlar** ile yazılmış.

**GO:** Hopkins + ikinci bağımsız tanısal ölçü uyumlu ve istikrarlı; Stage 3 clustering karşılaştırmasına geçilebilir.  
**REVIEW:** sonuç karmaşık veya temsil/encoding etkisi güçlü; duyarlılık analizini tamamla.  
**PIVOT:** anlamlı cluster structure bulunamadı; clustering'i ana contribution olmaktan çıkar, pattern/anomaly/prediction seçeneklerini değerlendir. **Olumsuz sonuçları saklama.**

## 7. Daha sonraki aşamalar: kapsam sınırı ve başarı ölçütleri

**Stage 3 / H3 — Clustering:** K-Means, K-Medoids, BIRCH, DBSCAN; gerekliyse başka metodoloji. Standardize edilmiş feature uzayında adil karşılaştırma; Silhouette (higher), Davies–Bouldin (lower), Calinski–Harabasz (higher), runtime, memory, stability (ör. ARI), cluster sizes, business interpretability. `Silhouette > 0.25` *zayıf ama incelenebilir*, `> 0.40` *umut verici* **öneri**; mutlak eşik değil. DBSCAN noise oranını ve noise handling'ini raporla. GO: kaliteli + nispeten stable + yorumlanabilir kümeler; tek metriğe göre şampiyon ilan etme.

**Stage 4 / H4 — PCA:** Explained variance ratio; 2D projection yalnızca görsel, gerekirse %75+ varyansı koruyan daha çok bileşen kullanılabilir. PCA=proof-of-cluster değildir.

**Stage 5 / H5 — Association mining:** Apriori vs FP-Growth (veya birinin farklı yapılandırmaları); discretization sadece training tarafında tanımlansın; min support/confidence/lift aday olarak sırasıyla `%1–2`, outcome prevalence'i aşan confidence, `lift > 1.2`. **Holdout** validasyonu ve çoklu test / küçük destek uyarısı şart. Nedensellik iddia etme.

**Stage 6 / H6 — Anomaly:** Isolation Forest vs Local Outlier Factor. 'Engagement Trap' (high engagement–poor commercial outcome) ve 'Quiet Converter' (low/medium engagement–high outcome) iş tanımıdır; IF/LOF kendiliğinden bu sınıfları üretmez. Etki büyüklüğü ve anlamlılık/süreklilik kontrolü; doğru baseline ile kıyasla.

**Stage 7 / H7a — Classification:** Trivial baseline + Logistic Regression/Random Forest/XGBoost veya uygulanabilir iki model. Pre-campaign features ve hedef tanımını dikkatle ayır; F1, ROC-AUC, PR-AUC, confusion matrix. `ROC-AUC > 0.70` sadece öneri; baseline'ı geçmek öncelikli. Split leakage, post-outcome feature ve repeated-creator contamination engellensin. Başlamadan seçilecek zaman/grup bazlı split stratejisi belgelenmeli.

**Stage 8 / H7b,H8 — Added value:** Raw/reach-engagement baseline vs engineered/multi-method representations; data-leakage'siz ablation. Tahmini `+3–5% F1/PR-AUC` **hedef**, garanti değil. Prediction artmasa bile descriptive insight yararlı olabilir; fakat fayda gerçek kullanım/holdout üzerinden dürüstçe değerlendirilir. Nihai paper trade-offs ve null bulguları açıklamalı.

## 8. Araştırma metodolojisi: vazgeçilmez sınırlar

1. **Dataset gerçekliği teyit edilmeden "real-life" diye yazma.** Kaggle platformu tek başına veri gerçekliğinin kanıtı değildir.
2. **Sales, conversion, revenue, ROI kavramlarını karıştırma.** Reach başına sales ölçümü, purchase conversion rate değildir. ROI için güvenilir gelir ve maliyet gerekir.
3. **Association != causation.** Başarılı platform/niche kombinasyonu başka değişkenlerle açıklanabilir.
4. **Target leakage ve circular validation yok.** Sales derived feature'ı sales prediction'a sokma. Outcome'u clustering feature'ı olarak kullanırsan aynı outcome ile kümeyi bağımsız dış validasyon olarak övme.
5. **İş problemine göre granularity sabit olsun.** Campaign-level hedef ile video-level veya interaction-level satırları birleştirirken duplicate/bias kontrolü yap.
6. **Train/test ayrımını deney başlamadan belirle.** Split mümkünse zaman veya creator/campaign grubu bazlı; preprocessing ve unsupervised transformation train split'te fit edilsin (predictive tasks).
7. **Statistical tests; effect size; uncertainty:** mümkün olduğunda confidence interval/bootstrap; küçük support kurallarını abartma.
8. **Privacy:** kişisel veri toplama/scraping yok; kamuya açık/anonim veri ve kaynak kullanım şartları içinde kal.
9. **Reproducibility:** seed=42 temel başlangıç, tüm paket sürümleri, konfigürasyonlar, veri kaynağı, dosya hash'i, çalışma komutları ve çıktı dizinleri kaydedilsin.
10. **Dataset araştırma sorusunu değiştirecekse açık karar gerektirir.** Satış verisi yoksa audience/content performance için scope değişimini belirgin bir `PIVOT` olarak belgele.
11. **Kod ve paper'da olmayan sonuçları uydurma.** Eşikler önceden planlanan operasyonel hedeflerdir, gerçekleşmiş bulgular değil.
12. **Notebooks yardımcı olabilir; ancak ana analiz yeniden çalıştırılabilir script/modül ile yürüsün.** Güçlü baseline tercih et; pipeline'ın doğruluğu dashboard'dan önemli.

## 9. Öncelikli repository yapısı (esnek, mevcut repoyu bozma)

```text
project-root/
├── PROJECT_CONTEXT.md                  # Bu belge
├── README.md                           # Çalıştırma ve durum
├── requirements.txt veya pyproject.toml
├── configs/
│   └── experiment.yaml
├── data/
│   ├── raw/                             # Git dışında
│   ├── interim/                         # Git dışında
│   └── processed/                       # Git dışında
├── src/
│   ├── data_audit.py
│   ├── preprocess.py
│   ├── features.py
│   ├── clustering_tendency.py
│   └── utils.py
├── tests/
│   ├── test_features.py
│   └── test_clustering_tendency.py
├── reports/
│   ├── dataset_decision.md
│   ├── stage_01_eda.md
│   ├── stage_02_clusterability.md
│   ├── stage_gates.md
│   ├── figures/
│   └── tables/
└── notebooks/                            # opsiyonel
```

İyi bir ilk veri hattı: kaynak dosya -> verified raw -> validated typed table -> features -> diagnostic reports -> GO/REVIEW/PIVOT kararı. Büyük dataset'i gereksiz yere Git'e commit etme. Gerekli paketler minimal olsun (ör. pandas, numpy, scikit-learn, scipy, matplotlib, pyarrow, pytest; dataset tipine göre ayarla).

## 10. Codex için ilk görevler ve "Definition of Done"

**Görev A — Repo discovery:** Mevcut dosyaları incele, yanlışlıkla silme/değiştirme yapma; çalışma ortamını ve mevcut veri dosyalarını bul. Varsa eski çalışmaların üzerine körlemesine yazma.

**Görev B — Dataset audit:** Yukarıdaki adayları gerçekten kontrol et; erişebildiğin dosyadan şema çıkar; sample gözlemle; metadata/license/provenance bul; `reports/dataset_decision.md` oluştur. Ulaşılamayan kaynağa **unverified** yaz; uydurma kolon üretme. ROI dataset synthetic çıkarsa otomatik 'real influencer sales' gibi sunma.

**Görev C — Stage 1:** Dataset seçimi GO olduysa data cleaning/EDA ve feature engineering uygula; mümkünse tekrar kullanılabilir komut dosyaları, data dictionary, testler, figürler ve `reports/stage_01_eda.md` hazırla. Kullanılabilir alanlar yetersizse gate raporunu yaz ve stop/pivot kararını açık belirt.

**Görev D — Stage 2:** Stage 1 uygunsa clustering tendency testlerini birden fazla seed ve null modeliyle uygula; Hopkins testini küçük sentetik oyuncak testlerle doğrula (test amacıyla synthetic kullanmak serbest; **ana araştırma dataset'i yerine geçmez**); `reports/stage_02_clusterability.md` oluştur.

**Görev E — Stage-gate raporu:** `reports/stage_gates.md` içinde her aşama için `NOT STARTED / RUNNING / GO / REVIEW / PIVOT`, kanıt dosyası, karar gerekçesi, blokaj ve sıradaki adım bulunmalı. Hiçbir deneyin sonucunu görmeden GO yazma.

**Görev F — Son mesaj:** Yapılan işleri, hangi dataset'in doğrulandığını, H1/H2'nin hangi kanıtlarla desteklendiğini veya desteklenmediğini, çıktı dosyalarını, komutları, eksik erişimleri ve sonraki tek mantıklı adımı kısa özetle.

**İlk çalışma sınırı:** Şimdilik Stage 3–8'i implement etme. Sadece Stage 0–2'yi mümkün olduğu ölçüde tamamla; zayıf/verisiz durumda daha ileriye geçmiş gibi gösterme.

## 11. Daha sonraki haftalar için yol haritası (yeniden planlanabilir)

- İlk dönem: dataset provenance + EDA + features + H2.
- Sonraki dönem: clustering karşılaştırması, evaluation, PCA.
- Sonra: association rules ve anomaly analyses.
- Zaman ve dataset izin verirse: prediction ve ablation.
- Son aşama: paper, grafikler, demo, sunum; deneyleri teslimden önce dondur.

### Paper için hedef bölümler

Abstract; Introduction & problem; Related Work; Data Provenance & Ethics; Research Questions and Hypotheses; Methods; Experimental Setup; Results; Comparison & Ablation; Limitations; Conclusion; References. **Yöntemlerin seçimi pazarlama araştırma sorularına bağlanmalı.**

## 12. Referans bağlantılar (doğrulama başlangıcı)

- Course background: kullanıcının Data Mining syllabus'ı ve Chapter 8 clustering slaytları (bu Codex chat'ine otomatik aktarıldığını varsayma; bu belge çalışma için yeterli özet sunar).
- Kaggle Influencer Marketing ROI: https://www.kaggle.com/datasets/tfisthis/influencer-marketing-roi-dataset
- SAGraph academic paper: https://arxiv.org/abs/2403.15105
- SAGraph repository/data instructions: https://github.com/xiaoqzhwhu/SAGraph/
- YouTube Trending Video Statistics: https://www.kaggle.com/datasets/datasnaek/youtube-new
- UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Google Analytics 4 sample data limitations: https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset

---

**Başarı tanımı:** İlk teslimde "influencer araştırmasının verisini doğruladık / hangi koşulla kullanabileceğimizi belirledik", "H1 için temiz feature ve EDA kanıtını ürettik", "H2 için istikrarlı, tekrar üretilebilir clustering-tendency tanısını yaptık" diyebilmeliyiz. Sonuç negatifse de güvenilir negatif sonuç olsun.
