---
title: "Bias and Fairness"
description: "Understanding sources of algorithmic bias and techniques for fairness mitigation."
aliases:
  - Algorithmic Bias
tags:
  - CS/artificial-intelligence
  - ai-ethics
---

> [!abstract] Abstract
> Algorithmic bias is a significant ethical challenge in data science, as it can perpetuate systemic inequalities. It often emerges when training data does not represent the diversity of the real world, causing models to favor certain demographic groups over others.

---
## Sources of Bias

- **Data Representation:** Underrepresentation of certain populations in training datasets.
- **Model Architecture:** Algorithms may amplify historical or societal biases.
- **Deployment Context:** Feedback loops and real-world usage can reinforce biased outcomes.

> [!example] Facial Recognition
> Facial recognition systems have shown lower accuracy for darker-skinned individuals due to biased training data.

---
## Mitigation Techniques

- **Adversarial Debiasing:** Introduce counterfactual examples to reduce bias.
- **Reweighting Samples:** Adjust training data distributions to improve fairness.
- **Fairness-Aware Loss Functions:** Penalize biased predictions during training.
- **Fairness Audits:** Use tools like IBM’s AI Fairness 360 or Google’s What-If Tool to evaluate model bias.

---
## Multidisciplinary Approaches & Monitoring

- **Ethics and Law:** Integrate legal and philosophical perspectives into model development.
- **Diverse Teams:** Encourage inclusive development environments to surface blind spots.
- **AI Ethics Boards:** Establish cross-functional review panels to assess fairness risks.
- **Production Oversight:** Track model behavior over time as data distributions shift.
- **Periodic Reevaluation:** Update models and retrain as needed to maintain fairness.
- **Regulatory Compliance:** Align with legislation such as the Algorithmic Accountability Act (U.S.).

---
## Case Study: Facial Recognition Bias

A 2018 study by MIT researcher Joy Buolamwini found that facial recognition systems from IBM, Microsoft, and Amazon had significantly higher error rates for darker-skinned individuals, with rates reaching 34.7% for dark-skinned women.

**Responses:**
- **IBM:** Released the Diversity in Faces dataset to improve representation.
- **Microsoft:** Introduced Fairlearn, an open-source bias mitigation toolkit.
- **Policy Action:** Both companies paused law enforcement sales pending ethical regulation.

---
## Video Resource

![](https://www.youtube.com/embed/wmyVODy_WD8?feature=oembed)

---
# Related Notes

- [[Artificial Intelligence/Ethical AI and Policy/index|Ethical AI and Policy]]