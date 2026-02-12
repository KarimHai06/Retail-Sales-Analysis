# Retail Sales Analytics: Trends, Product Mix, Customers, Geography & Logistics
## Project Goal: 
  This project delivers an end-to-end business analytics case study using a retail “Superstore” transactions dataset. The objective is to identify __revenue drivers, seasonality, product concentration (Pareto 80/20), customer contribution patterns, geographic imbalances,__ and __logistics performance__, and translate the findings into __actionable recommendations.__

## Key Questions

1. ### Sales Dynamics

- How does revenue change over time (monthly / yearly)?

- Are there seasonal peaks or consistent trends?

2. ### Product Performance

- Which categories and sub-categories generate the most sales?

- Which sub-categories form the core revenue base (Pareto 80/20)?

- Are any sub-categories declining over time?

3. ### Customer Analysis

- How do customer segments (Consumer / Corporate / Home Office) contribute to sales?

- Who are the key customers and what do they buy?

4. ### Geographic Analysis

- How do regions, states, and cities differ in sales?

- Where do unusual “category dominance” patterns appear and what might they imply?

5. ### Logistics

- How does shipping mode affect delivery time?

- Does shipping mode affect order value or sales distribution?

## Dataset

- Source: Kaggle (Superstore Sales Dataset) 

- Period: __2015–2018__

- Size: __9,800 rows, 18 columns__

- Granularity: order-line level transactions (each row represents a purchased item/line in an order)

## Tech Stack

1. __Python__ (Pandas, NumPy)

2. __Plotly__ (interactive visualizations)

3. Feature engineering for time-series and logistics analysis

## Data Preparation and Quality Checks
### 1) Initial Data Audit

  During the initial audit:
 
   - The dataset contained __18 columns__ and __9,800 rows.__

   - _Postal Code_ had __11 missing values.__

   - Duplicate-like entries were identified: __8 rows__ shared the same key fields and were identical except for _Sales_.

   - _Order Date_ and _Ship Date_ were not in the correct data type for time analysis (strings rather than datetime).

### 2) Missing Values: _Postal Code_

   - 11 missing values were found in _Postal Code_.

   - All missing records corresponded to the same geographic area (Vermont/Burlington context).

   - Missing postal codes were imputed as __5401__, which corresponds to __ZIP 05401 (Burlington, Chittenden County).__
  Note: ZIP codes in this region often start with a leading zero (e.g., _05401_). If stored as      numeric, that leading zero may be dropped, producing _5401_.

### Why impute instead of dropping rows?

- Missingness was extremely small (~0.1%).

- Keeping these rows preserves revenue and avoids unnecessary data loss.

- Imputation improves consistency for geographic slicing.

### 3) Consolidation of Split Line-Items (Duplicate-like Records)

A duplicate check identified __8 transactions__ that were:

- identical across all descriptive fields (order, customer, product, location, etc.)

- different only in the _Sales_ value

These were interpreted as __split line-items__ (multiple entries representing the same product in the same order, split across records). To maintain correct totals while avoiding overstated line counts:

- All fields except _Row_ID_ and _Sales_ were used as grouping keys.

- _Sales_ values were summed.

- A clean sequential _Row_ID_ was rebuilt after consolidation.

This step ensures:

- revenue remains accurate

- analysis (especially order- and product-level aggregation) is cleaner and more interpretable

### 4) Date Standardization

Because dates were stored in a day-first format (e.g., _15/04/2018_), both date fields were converted using:

- _dayfirst=True_

This enabled:

- monthly and yearly grouping

- seasonality analysis

- delivery-time calculation (_Ship Date − Order Date_)

## Analysis and Findings
### 1) Sales Dynamics (Trend + Seasonality)
### Key Findings

- Sales show a __clear long-term growth trend__, especially from __2017 onward__.

- __Seasonality is strong__ with recurring peaks and troughs across years.

### Seasonality: Strong Months

Several months consistently show stronger sales:

- __March__: appears as a notable increase. Based on pattern inspection, it likely reflects rebound after a weak February rather than a standalone “March effect.”

- __September (and late August)__: rising demand aligns with back-to-school/education-related purchasing cycles.

- __November__: the most pronounced peak, consistent with __Black Friday promotions and holiday pre-shopping.__

### Low Sales Periods

- __February__ is consistently the weakest month, likely due to:

  - fewer calendar days

  - post-holiday demand decline

- __October__ can visually appear as a dip, but in absolute terms sales remain above annual average; it behaves more like a stabilization before the November spike.

### Year-over-Year Trend

- Strong growth is visible in the last two years:

  - __~25%__ increase in 2017

  - __~20%__ increase in 2018

- This indicates a positive long-term trajectory potentially driven by:

  - customer base growth

  - assortment expansion

  - business scaling effects

### 2) Product & Category Performance
### Category-Level

- All three categories (Technology, Furniture, Office Supplies) contribute meaningfully to revenue.

- __Technology__ leads in total sales, indicating stronger demand for tech-oriented products.

### Category Trends Over Time

- The most visible growth in 2017–2018 is primarily driven by:

  - __Technology__

  - __Office Supplies__

- Furniture grows more slowly relative to the other two.

### Pareto Analysis: 80/20 Sub-Categories

A Pareto (80/20) analysis shows that roughly __80% of revenue is generated by a small set of sub-categories__, specifically:

- Phones

- Chairs

- Storage

- Tables

- Binders

- Machines

- Accessories

- Copiers

All other sub-categories contribute substantially less and are not primary revenue drivers.

### Declining Sub-Categories: Machines & Tables

Despite being among the core revenue drivers, __Machines__ and __Tables__ show __negative dynamics__ over time:

- strong performance in 2015

- visible declines particularly in 2016 and 2018

This pattern signals risk:

- these sub-categories may lose importance as drivers

- they require deeper follow-up (pricing, competitiveness, lifecycle effects)

### 3) Customer Analysis (Segment + Revenue Concentration)
### Customer Segments

The dataset includes three segments:

- Consumer

- Corporate

- Home Office

Customers were grouped into contribution “classes” inside each segment and category:

- __Class 1 (0–20%)__: customers generating the first 20% of sales

- __Class 2 (20–80%)__: core customer base

- __Class 3 (80–100%)__: long tail / lowest contribution

### Stable Demand Sub-Categories

Across all segments and contribution classes, several sub-categories show stable demand:

- Technology / Phones

- Furniture / Chairs, Tables

- Office Supplies / Storage, Binders, Paper

- Appliances

These products appear “baseline” and are purchased broadly across customer types.

### High-Value (Class 1) Behavior

- High-value customers concentrate spending in more expensive sub-categories.

- __Copiers__ stands out as a premium sub-category driven heavily by high-value customers.

- Business implication: target Class 1 with premium offerings + service bundles.

### Mid/Low (Classes 2–3) Behavior

- More frequent purchases of inexpensive supporting items.

- __Accessories__ is especially strong here, making it ideal for broad promotions and cross-selling.

### Notes on Machines

Machines have lower purchase frequency in a 4-year window—likely due to longer replacement cycles. Low frequency does not imply low importance. A practical business approach:

- budget machine lines for mass replacement

- premium machine lines for high-requirement corporate buyers

### 4) Geographic Analysis (Region → State → City)
### Regions

- __West__ is the highest-performing region by total sales.

- The category mix by region is fairly consistent:

  - roughly balanced across the three categories

  - slight tilt toward Technology

- Regional differences appear driven by __scale__, not by major structural differences.

### States: Category Dominance (>50%)

Some states show extreme concentration where one category contributes __more than 50%__ of sales. This can signal:

- local specialization

- or missing/limited assortment in other categories

States with dominant __Technology__:

- Florida, Indiana, Delaware, Montana, Maine

States with dominant __Furniture__:

- Wisconsin, Vermont, Iowa, Idaho, Wyoming (likely influenced by low population scale), West Virginia

States with dominant __Office Supplies__:

- Georgia, Minnesota, Kansas, Missouri, North Dakota

Recommendation: follow-up assortment review in these states to confirm whether dominance reflects demand preference or supply constraints.

### Cities (Sales > $10,000)

To focus on meaningful markets, the analysis included cities with total sales above __$10,000.__

Notable imbalances:

- __East__: Newark, Lakewood (Technology-heavy); Columbia (low Technology share)

- __West__: generally balanced; San Diego (Office Supplies-heavy)

- __South__: most imbalanced region

  - Jacksonville, Arlington, Charlotte, Burlington (Technology-heavy)

  - Atlanta, Richmond (Office Supplies-heavy)

- __Central__:

  - Chicago, San Antonio, Lafayette (Technology-heavy)

  - Detroit, Minneapolis, Milwaukee (low Technology share)

  - Minneapolis (Office Supplies-heavy)

  - Milwaukee (Furniture-heavy)

  - Springfield (low Furniture share)

### Overall interpretation
These category skews are more consistent with __assortment imbalance__ than absence of demand. Customers allocate spending toward available categories; the global mix remains balanced. This suggests growth potential via assortment expansion in specific markets.

### 5) Logistics Analysis (Ship Mode + Delivery Time)
### Delivery Time

Shipping mode strongly impacts delivery time:

- Standard Class: ~5 days

- Second Class: ~3 days

- First Class: ~1 day

- Same Day: within 24 hours

### Customer Preference

- Standard Class dominates (~59% of all sales/orders).

- Faster options are less popular, potentially reflecting shipping price sensitivity.

### Impact on Order Value

- No meaningful difference in average order value across shipping modes.

- Shipping choice appears related to __delivery preference__, not order importance/value.

### Segment / Category Differences

- Segment does not materially influence ship mode choice.

- Corporate buyers use Same Day slightly more, but not enough to be a major differentiator.

- Technology uses faster shipping slightly more often, but without strong dominance.


## Final Recommendations

### 1. Double down on Technology growth
Focus on Technology category and core revenue sub-categories: Phones, Chairs, Copiers.

### 2. Target high-value customers with premium bundles
Copiers are heavily dependent on top customers—offer premium service contracts, upgrades, and tailored deals.

### 3. Scale Accessories for mass-market segments
Accessories perform well among mid/low contribution customers; use promotions and cross-sell strategies.

### 4. Fix assortment gaps in skewed states/cities
Where one category dominates >50%, investigate whether other categories are underrepresented and expand accordingly.

### 5. Plan marketing around two peak seasons

- Aug–Sep (back-to-school)

- Nov–Dec (holiday + promotions)

### 6. Stimulate February demand
Run targeted February campaigns: bundles, discounts on consumables, and loyalty incentives.


Code adapted from "General Analysis in Russian" (Kaggle) by "Глеб Мехряков"
