# Dataset Selection and Provenance

**Status: undecided. Do not assert a selected dataset until verified.**

## Selection criteria
1. Binary classification with explicit semantic positive class and ethically appropriate usage.
2. Manageable on CPU; ideally several hundred to a few thousand independent records.
3. Numeric features or feasible categorical handling compatible with GaussianNB and LogisticRegression.
4. Published source/attribution/license or access terms and stable retrieval steps.
5. Genuine analytical value: class overlap, feature correlations, missing/invalid values, imbalance, etc. (not all are mandatory).
6. No direct leakage feature, impossible-to-obtain-at-inference variable, duplicated sample leakage, or unjustified post-outcome information.

## Candidates to verify
| Candidate | Why consider | Caution |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | Clean numeric, binary labels, GaussianNB vs Logistic Regression assumptions | Limited genuine cleaning; avoid any medical conclusions |
| Titanic survival | Missing values, mixed feature types, interpretable preprocessing | Mixed categorical features may require modelling decisions |
| Pima Indians Diabetes | Numeric, placeholder-zero analysis can matter | Check values, origin, restrictions and fairness of interpretation |
| Banknote authentication | Numeric, accessible classroom demo | Possibly little cleaning effort |
| Bank marketing | Realistic, potential imbalance and categoricals | Potentially more complex/time consuming |

## Dataset decision record — fill after investigation
- Dataset name and exact version:
- Official source URL, download path, accessed date:
- License/usage permission:
- Citation or attribution:
- Number of rows / features:
- Target name, positive label and explanation:
- Feature types / possible predictors unavailable at inference:
- Missing / placeholder / duplicate / invalid records:
- Class counts and ratio:
- Independence, group or temporal structure affecting split:
- Risks / sensitive attributes / limits:
- Exact retrieval command, checksum if stored:
- Reason for choosing over alternatives:
- Approved by user? (if a preference or ambiguous tradeoff needs approval):

Do not automatically remove duplicates: legitimate repeated feature rows with different labels can reflect aleatoric uncertainty. Investigate row-level identity and timing.
