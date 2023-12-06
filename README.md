# Netflix Website Optimization Using A/B Testing

## Overview

This project presents a comprehensive A/B experimental analysis focused on optimizing the Netflix website's user interface. Our primary objective was to identify design features that minimize users' average browsing time, thereby enhancing user experience and reducing choice paralysis.


## Experiment Design

We targeted four key design features of the Netflix website:
1. **Tile Size:** The ratio of the tile height to the overall screen height.
2. **Match Score:** The probability, expressed as a percentage, that a user would enjoy a show or movie based on their viewing history.
3. **Preview Length:** The duration in seconds of the show or movie’s preview.
4. **Preview Type:** The type of preview shown to the users (actual content or a short teaser trailer).

## Methodology

### 2k Factorial Analysis
To understand the effect of each design factor on browsing time, we utilized a 2k factorial design. The general model for the factorial design is given by:
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + ... + ε


### Response Surface Optimization
For finding the approximate optimum values of significant factors, we used the response surface methodology. The quadratic model used is:
Y = β₀ + ΣβᵢXᵢ + ΣβᵢᵢXᵢ² + ΣβᵢⱼXᵢXⱼ + ε

### Pairwise T-Test Grid Search
We performed a pairwise t-test grid search to improve the accuracy of the optimum value predictions.

## Key Findings

- **Tile Size:** No significant impact on user browsing time.
- **Optimum Values:**
  - Match Score: 75%
  - Preview Length: 74 seconds
  - Preview Type: Teaser trailer preview

## Limitations and Future Work

- Limited number of condition combinations tested due to resource constraints.
- A more extensive pairwise grid search could further refine the optimum condition set.
- Larger user groups could enhance the robustness of results.

## Conclusion

The study concludes that the tile size does not affect browsing time, whereas specific values for match score, preview length, and preview type significantly enhance user experience by reducing browsing time.
