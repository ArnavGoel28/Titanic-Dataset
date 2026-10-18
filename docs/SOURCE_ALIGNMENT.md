# Alignment to Teacher-Provided Reading

## Direct assignment and reference grounding
- **Teacher's assignment (as described by student):** any dataset, necessary preprocessing/cleaning/sampling, Naive Bayes and Logistic Regression, results analysis/validation. Specific dataset, metrics, report format and GitHub workflow are NOT prescribed in the assignment wording provided.
- **Generative and Discriminative Classifiers (11 pages):** worked binary student classification example; Gaussian naive Bayes class-conditionals and conditional independence; maximum likelihood estimates of class proportions, means and variances; sigmoid log-odds and decision boundaries; logistic regression Bernoulli likelihood, binary cross-entropy, gradient descent; correlated/duplicate feature caveats and model comparison.
- **Transforming Densities, Measuring Information (19 pages):** change-of-variables/Jacobian; entropy, maximum entropy, KL divergence, cross-entropy, conditional entropy, mutual information and links to density modelling; supporting Gaussian/probability material. Many topics are context, NOT mandatory operations in the empirical assignment.
- **Bishop & Bishop, Deep Learning, Chapters 1–5 (user supplied):** generalization/overfitting (Ch1); probability, Bayes and likelihood (Ch2); common distributions and Gaussian estimation (Ch3); regression/least-squares foundations (Ch4); classification, discriminant functions, generative/discriminative probabilistic classifiers (Ch5).

## Theory to explain in the experiment
1. Gaussian Naive Bayes estimates `P(y)` and class-conditional Gaussian feature densities and combines them through Bayes' rule.
2. Logistic Regression directly estimates `P(y=1|x)` using a sigmoid of a weighted feature sum; fit with Bernoulli negative log-likelihood / binary cross-entropy.
3. The decision rule (often threshold 0.5) must be separated from probability modelling; threshold effects matter for precision/recall.
4. Correlated predictors can violate naive conditional independence; small datasets can produce uncertain/overconfident estimates; report observed results without universal conclusions.
5. Cross-entropy/log-loss measures predicted probabilities, whereas accuracy/F1 depend on thresholded predictions.

## Boundaries
Do not claim the teacher mandates PCA, SMOTE, neural nets, calibration, Bayes-factor calculations, density Jacobians or specific metrics. These are options for scientific quality or contextual discussion, not confirmed requirements.
