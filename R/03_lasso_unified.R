library(tidyverse)
library(glmnet)
library(ggplot2)

panel <- read_csv("data/processed/monthly_panel.csv") %>%
  mutate(date = as.Date(date))

macro_vars     <- c("michigan_inflation", "cleveland_inflation", "spf_cpi_forecast",
                    "unemployment", "ppi", "oil_wti", "pce_yoy")
market_vars    <- c("kalshi_implied_cpi")
sentiment_vars <- c("gdelt_tone", "google_trends_inflation")
all_pred_vars  <- c(macro_vars, market_vars, sentiment_vars)

category_map <- c(
  michigan_inflation      = "Macro/Survey",
  cleveland_inflation     = "Macro/Survey",
  spf_cpi_forecast        = "Macro/Survey",
  unemployment            = "Macro/Survey",
  ppi                     = "Macro/Survey",
  oil_wti                 = "Macro/Survey",
  pce_yoy                 = "Macro/Survey",
  kalshi_implied_cpi      = "Prediction Market",
  gdelt_tone              = "Sentiment",
  google_trends_inflation = "Sentiment"
)

lasso_data <- panel %>%
  select(cpi_yoy_next, all_of(all_pred_vars)) %>%
  drop_na()

X <- lasso_data %>% select(-cpi_yoy_next) %>% as.matrix() %>% scale()
y <- lasso_data$cpi_yoy_next

set.seed(42)
cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = nrow(X))

cat(sprintf("Lambda min : %.4f\nLambda 1se: %.4f\n",
            cv_fit$lambda.min, cv_fit$lambda.1se))

coefs   <- coef(cv_fit, s = "lambda.1se")
coef_df <- tibble(
  Variable    = rownames(coefs)[-1],
  Coefficient = as.numeric(coefs)[-1]
) %>%
  filter(Coefficient != 0) %>%
  mutate(
    Category = category_map[Variable],
    Variable = fct_reorder(Variable, abs(Coefficient))
  )

lasso_pred <- as.numeric(predict(cv_fit, X, s = "lambda.1se"))
lasso_rmse <- sqrt(mean((y - lasso_pred)^2))
lasso_r2   <- 1 - sum((y - lasso_pred)^2) / sum((y - mean(y))^2)

cat(sprintf("\nLASSO at lambda.1se: R2 = %.3f  RMSE = %.3f\n", lasso_r2, lasso_rmse))
cat("Non-zero coefficients:\n")
print(coef_df)

write_csv(coef_df, "output/tables/lasso_results.csv")
write_csv(
  tibble(model = "Unified LASSO", R2 = lasso_r2, RMSE = lasso_rmse,
         N = nrow(lasso_data), n_nonzero = nrow(coef_df)),
  "output/tables/lasso_metrics.csv"
)

if (nrow(coef_df) > 0) {
  p2 <- ggplot(coef_df, aes(x = Variable, y = Coefficient, fill = Category)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = c(
      "Macro/Survey"      = "#2196F3",
      "Prediction Market" = "#FF5722",
      "Sentiment"         = "#4CAF50"
    )) +
    labs(
      title    = "LASSO: Non-Zero Coefficients by Category",
      subtitle = sprintf("lambda.1se  |  R2 = %.3f  |  RMSE = %.3f", lasso_r2, lasso_rmse),
      x = NULL, y = "Standardized Coefficient",
      fill = "Category"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")

  ggsave("output/figures/fig2_lasso_coefs.png", p2, width = 8, height = 5, dpi = 150)
  cat("Figure 2 saved: output/figures/fig2_lasso_coefs.png\n")
} else {
  cat("No non-zero coefs at lambda.1se — try lambda.min instead.\n")
}
