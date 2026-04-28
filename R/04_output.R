library(tidyverse)
library(ggplot2)

ols   <- read_csv("output/tables/ols_results.csv")
lasso <- read_csv("output/tables/lasso_metrics.csv") %>%
  select(Model = model, R2, RMSE, N)

ols_slim   <- ols %>% select(Model, N, R2, RMSE)
comparison <- bind_rows(ols_slim, lasso) %>%
  mutate(Model = factor(Model, levels = c(
    "Macro/Surveys OLS", "Prediction Markets OLS",
    "Sentiment OLS",     "Unified LASSO"
  )))

print(comparison)
write_csv(comparison, "output/tables/model_comparison.csv")
cat("Model comparison table saved: output/tables/model_comparison.csv\n")

fill_colors <- c("#2196F3", "#FF5722", "#4CAF50", "#9C27B0")

p3 <- comparison %>%
  drop_na(R2) %>%
  mutate(fill_col = fill_colors[seq_len(n())]) %>%
  ggplot(aes(x = Model, y = R2, fill = fill_col)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("R2=%.3f\nRMSE=%.3f", R2, RMSE)),
            vjust = -0.3, size = 3.5) +
  scale_fill_identity() +
  scale_x_discrete(labels = function(x) str_wrap(x, 12)) +
  ylim(0, 1.15) +
  labs(
    title    = "Model Comparison: In-Sample R2 and RMSE",
    subtitle = "Predicting next-month CPI YoY (Jan 2022-Mar 2026)",
    x = NULL, y = "R2"
  ) +
  theme_minimal(base_size = 12)

ggsave("output/figures/fig3_comparison.png", p3, width = 8, height = 5, dpi = 150)
cat("Figure 3 saved: output/figures/fig3_comparison.png\n")
