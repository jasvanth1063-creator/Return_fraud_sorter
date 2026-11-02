## Findings (numbers are from the tables above, training data, 43,068 returns, base abuse rate 7.8%)

### 1. The target
- 7.8% abusive, 48.1% fit, 28.0% quality, 16.1% wrong item. Approving everything is 92% "accurate" and catches nothing, so judge models by cost and ranking quality, never accuracy.
- The abusive share is **not constant**: it moves between 5.5% (April 2026) and 10.0% (December 2025), with peaks in December and March. A model must not assume one fixed rate, and calibration should be checked on recent months.

### 2. Strongest signals (what abusive returns look like)
| Signal | Abuse rate | Compared with 7.8% base |
|---|---|---|
| Return requested 22-30 days after delivery | 22.7% | 2.9x |
| Return requested 15-21 days after delivery | 13.2% | 1.7x |
| Return requested 2-7 days after delivery | 3.3% | 0.4x |
| Customer already has **one earlier abusive return** | 43.8% | 5.6x |
| Customer has two or more earlier abusive returns | 73.5-76.2% | about 9.5x |
| Same exact reason text reused by the same customer | 32.1% | 4.1x |
| Prior return rate above 0.75 | 15.4% | 2.0x |
| Late (>14 days) + cash on delivery + single item | 27.5% | 3.5x |
| Single-item order | 11.1% | 1.4x (multi-item orders: 3.4-4.8%) |
| Cash on delivery | 10.6% | 1.4x (prepaid: 5.5-6.3%) |
| Discount above 50% | 12.4% | 1.6x |
| Item price above 2,199 INR | 19.5% | 2.5x |
| Men Ethnic category | 17.5% | 2.2x (sherwani 29.5%, nehru jacket 26.8%, kurta pyjama set 26.7%) |
| Free size / semi-stitched items | 17.0% / 17.6% | 2.2x |
| Velvet / jacquard / art silk / chiffon fabrics | 19-28% | 2.5-3.6x |
| Reason text claims an empty box | 28.0% | 3.6x |

Customer history is the richest evidence: 324 customers with three or more abusive returns account for **48.8% of all abusive returns**, and customers with 8+ returns average 1.4 abusive returns each.

### 3. Weak signals and traps
- **Delivery speed, warehouse and city tier** barely move the abuse rate (7.2-8.4% across their levels).
- **Order value is not fraud evidence**: 7.5% below 1,000 INR vs 8.0% above. Item *price* matters at the extremes, but the pattern is U-shaped (cheapest band 8.8%, mid bands 4-5%, top band 19.5%).
- **New customers are not riskier.** Customers with 30 days or less of tenure are abusive only 3.3% of the time, and first-order customers (no history) 7.5%, about the base rate. Missing history must not be punished.
- Univariate ranking power (AUC, 0.5 = useless): days since delivery 0.68, prior return rate 0.63, price 0.61, discount 0.55, everything else about 0.5.
- **Text is weak evidence of abuse.** Language mix and length are identical across labels (non-English share 42-45% for every class). Abusers copy genuine wording ("size issue", "cloth is thin"). Text separates the *genuine reasons* well (e.g. "too small" in 31% of fit returns and only 10% of abusive ones), so it is best used to explain the genuine reason and to flag *inconsistencies*, not as a fraud detector.
- Some phrases occur only in genuine returns (e.g. "extra", "returning", "sure", "mangaye", "isliye" - zero abusive occurrences in hundreds of rows). These are strong "this is genuine" cues and suggest part of the genuine text follows templates.

### 4. Fairness
- Abuse base rates are close between groups: new customers 6.3% vs 8.1% others, low-value orders 7.5% vs 8.0%, non-English text 8.0% vs 7.7%. There is no legitimate reason to flag one group more often.
- **Proxy risk:** new customers use cash on delivery 57% of the time (all orders: 39%), and low-value orders are single-item 85% of the time (all orders: 50%). A model that leans on COD or single-item status can end up flagging these groups more often, even though the groups are not riskier. This must be measured, and fixed without flagging low-risk customers just to balance numbers.
- Language must not be a model input: it is a fairness group and could become a proxy.

### 5. Train vs test
- The test period (July-September 2026) follows the training period. Categorical mixes are stable (population stability index below 0.004), there are **no products that appear only in the test set**, and 73.7% of test requests come from customers who also appear in train.
- Mild shifts: discount +15% (27.7 to 31.9), prior orders +16%, tenure +11%, item price -7%. Nothing severe, but it argues for time-based validation and for checking calibration on the most recent months.

### 6. Fit returns and the size chart
- Five products clearly lead the genuine fit-return counts: **176, 138, 135, 124 and 117** returns. The sixth has 68 and the seventh 67, so the top 5 is well separated.
- They are two kurta/kurta-set items, two denim jeans and one formal shirt. Across all products, kurtas (4,572), jeans (2,984) and kurta sets (2,082) lead fit complaints.
- Fit complaints are 31% "too small", 21% "too large" and the rest unspecific.
- The data lists returns, not all orders, so a true fit-return *rate* per product cannot be computed; ranking is by volume of genuine fit returns.

### 7. What this means for modelling
1. Build **point-in-time customer history** (earlier returns, earlier outcomes, repeated reasons, return delay) - it carries most of the signal.
2. Use timing, payment, basket size, price, discount and category as supporting features; use text for the genuine reason and consistency checks.
3. Validate **by time** and group by customer, because customers recur across the split.
4. Calibrate probabilities on recent months, since the base rate drifts between 5.5% and 10%.
5. Audit fairness for new, low-value and non-English customers and avoid proxy-driven flagging.
