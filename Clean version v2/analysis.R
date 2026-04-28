# CPI Prediction: Macro/Surveys vs. Prediction Markets vs. Sentiment
# ECON 460 — USC Spring 2026
#
# Input:  data/processed/monthly_panel.csv  (built by the two Jupyter notebooks)
# Output: output/figures/   — three PNG plots
#         output/tables/    — OLS summary, LASSO coefficients, model comparison, stargazer HTML
#
# Run from the project root:  Rscript analysis.R

# ── 0. Packages ────────────────────────────────────────────────────────────────
pkgs <- c("glmnet", "tidyverse", "ggplot2", "stargazer",
          "lubridate", "readr", "dplyr", "tidyr", "scales", "gridExtra")
new  <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(tidyverse); library(glmnet); library(ggplot2); library(stargazer); library(gridExtra)
})

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables",  recursive = TRUE, showWarnings = FALSE)

# ── 1. Load panel ──────────────────────────────────────────────────────────────
panel <- read_csv("data/processed/monthly_panel.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date))

cat(sprintf("Panel loaded: %d rows x %d cols  (%s to %s)\n",
            nrow(panel), ncol(panel),
            as.character(min(panel$date)), as.character(max(panel$date))))

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

cat_colors <- c("Macro/Survey" = "#2196F3", "Prediction Market" = "#FF5722", "Sentiment" = "#4CAF50")

# ── 2. Category-Level OLS ──────────────────────────────────────────────────────
cat("\n── OLS Models ──\n")

# Standardize all predictors (z-score) so coefficients are comparable
panel_std <- panel %>%
  mutate(across(all_of(all_pred_vars), ~ as.numeric(scale(.))))

rmse <- function(fit) sqrt(mean(residuals(fit)^2))

fit_macro <- lm(
  cpi_yoy_next ~ .,
  data = panel_std %>% select(cpi_yoy_next, all_of(macro_vars)) %>% drop_na()
)
fit_market <- lm(
  cpi_yoy_next ~ .,
  data = panel_std %>% select(cpi_yoy_next, all_of(market_vars)) %>% drop_na()
)
fit_sentiment <- lm(
  cpi_yoy_next ~ .,
  data = panel_std %>% select(cpi_yoy_next, all_of(sentiment_vars)) %>% drop_na()
)

ols_results <- tibble(
  Model  = c("Macro/Surveys OLS", "Prediction Markets OLS", "Sentiment OLS"),
  N      = c(nobs(fit_macro),  nobs(fit_market),  nobs(fit_sentiment)),
  R2     = round(c(summary(fit_macro)$r.squared,
                   summary(fit_market)$r.squared,
                   summary(fit_sentiment)$r.squared), 3),
  Adj_R2 = round(c(summary(fit_macro)$adj.r.squared,
                   summary(fit_market)$adj.r.squared,
                   summary(fit_sentiment)$adj.r.squared), 3),
  RMSE   = round(c(rmse(fit_macro), rmse(fit_market), rmse(fit_sentiment)), 3)
)

print(ols_results)
write_csv(ols_results, "output/tables/ols_results.csv")
cat("Saved output/tables/ols_results.csv\n")

# Stargazer HTML table (copy into written report)
stargazer(
  fit_macro, fit_market, fit_sentiment,
  type           = "html",
  title          = "Table 1: Category-Level OLS — Predicting Next-Month CPI YoY",
  column.labels  = c("Macro/Surveys", "Prediction Markets", "Sentiment"),
  dep.var.labels = "CPI YoY (t+1)",
  digits         = 3,
  no.space       = TRUE,
  out            = "output/tables/ols_stargazer.html"
)
cat("Saved output/tables/ols_stargazer.html\n")

# ── 3. Unified LASSO (all features compete, LOO-CV) ───────────────────────────
cat("\n── Unified LASSO ──\n")

lasso_data <- panel %>%
  select(cpi_yoy_next, all_of(all_pred_vars)) %>%
  drop_na()

X <- lasso_data %>% select(-cpi_yoy_next) %>% as.matrix() %>% scale()
y <- lasso_data$cpi_yoy_next

set.seed(42)
cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = nrow(X))  # LOO-CV

cat(sprintf("Lambda min : %.4f\nLambda 1se : %.4f\n", cv_fit$lambda.min, cv_fit$lambda.1se))

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

cat(sprintf("LASSO (lambda.1se): R2 = %.3f  RMSE = %.3f  Non-zero coefs: %d\n",
            lasso_r2, lasso_rmse, nrow(coef_df)))
print(coef_df)

write_csv(coef_df, "output/tables/lasso_results.csv")
write_csv(
  tibble(Model = "Unified LASSO", N = nrow(lasso_data),
         R2 = round(lasso_r2, 3), RMSE = round(lasso_rmse, 3),
         n_nonzero = nrow(coef_df)),
  "output/tables/lasso_metrics.csv"
)
cat("Saved output/tables/lasso_results.csv and output/tables/lasso_metrics.csv\n")

# ── 4. Model Comparison Table ─────────────────────────────────────────────────
cat("\n── Model Comparison ──\n")

lasso_row <- tibble(Model = "Unified LASSO",
                    N     = nrow(lasso_data),
                    R2    = round(lasso_r2, 3),
                    Adj_R2 = NA_real_,
                    RMSE  = round(lasso_rmse, 3))

comparison <- bind_rows(ols_results, lasso_row)
print(comparison)
write_csv(comparison, "output/tables/model_comparison.csv")
cat("Saved output/tables/model_comparison.csv\n")

# ── 5. Figure 1: CPI Time Series Overlay ─────────────────────────────────────
cat("\n── Figures ──\n")

ts_data <- panel %>%
  select(date, `Actual CPI YoY` = cpi_yoy,
         `Michigan Survey`       = michigan_inflation,
         `SPF Forecast`          = spf_cpi_forecast,
         `Kalshi Implied CPI`    = kalshi_implied_cpi) %>%
  pivot_longer(-date, names_to = "series", values_to = "value") %>%
  drop_na()

p1 <- ggplot(ts_data, aes(x = date, y = value, color = series, linetype = series)) +
  geom_line(linewidth = 0.9) +
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y") +
  scale_color_manual(values = c(
    "Actual CPI YoY"   = "black",
    "Michigan Survey"  = "#2196F3",
    "SPF Forecast"     = "#1565C0",
    "Kalshi Implied CPI" = "#FF5722"
  )) +
  labs(title    = "CPI YoY vs. Prediction Sources (Jan 2022 – Mar 2026)",
       x = NULL, y = "Percent (%)", color = NULL, linetype = NULL) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("output/figures/fig1_timeseries.png", p1, width = 10, height = 5, dpi = 150)
cat("Saved output/figures/fig1_timeseries.png\n")

# ── Figure 2: LASSO Coefficient Bar Chart ────────────────────────────────────
if (nrow(coef_df) > 0) {
  p2 <- ggplot(coef_df, aes(x = Variable, y = Coefficient, fill = Category)) +
    geom_col() +
    coord_flip() +
    scale_fill_manual(values = cat_colors) +
    labs(title    = "LASSO: Non-Zero Coefficients by Category",
         subtitle = sprintf("lambda.1se  |  R2 = %.3f  |  RMSE = %.3f", lasso_r2, lasso_rmse),
         x = NULL, y = "Standardized Coefficient", fill = "Category") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")

  ggsave("output/figures/fig2_lasso_coefs.png", p2, width = 8, height = 5, dpi = 150)
  cat("Saved output/figures/fig2_lasso_coefs.png\n")
} else {
  cat("No non-zero coefs at lambda.1se — consider using lambda.min.\n")
}

# ── Figure 3: Model Comparison (R² and RMSE) ─────────────────────────────────
model_levels <- c("Macro/Surveys OLS", "Prediction Markets OLS",
                  "Sentiment OLS",     "Unified LASSO")
comp_long <- comparison %>%
  filter(!is.na(R2)) %>%
  mutate(Model = factor(Model, levels = model_levels))

model_fill <- c("Macro/Surveys OLS"       = "#2196F3",
                "Prediction Markets OLS"  = "#FF5722",
                "Sentiment OLS"           = "#4CAF50",
                "Unified LASSO"           = "#9C27B0")

p3a <- ggplot(comp_long, aes(x = Model, y = R2, fill = Model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", R2)), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = model_fill) +
  scale_x_discrete(labels = function(x) str_wrap(x, 12)) +
  ylim(0, min(1.15, max(comp_long$R2, na.rm = TRUE) * 1.35)) +
  labs(title = "In-Sample R²", x = NULL, y = "R²") +
  theme_minimal(base_size = 11)

p3b <- ggplot(comp_long, aes(x = Model, y = RMSE, fill = Model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", RMSE)), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = model_fill) +
  scale_x_discrete(labels = function(x) str_wrap(x, 12)) +
  labs(title = "In-Sample RMSE (lower = better)", x = NULL, y = "RMSE (pp)") +
  theme_minimal(base_size = 11)

p3 <- arrangeGrob(p3a, p3b, ncol = 2,
                  top = "Model Comparison: Predicting Next-Month CPI YoY (Jan 2022 – Mar 2026)")
ggsave("output/figures/fig3_model_comparison.png", p3, width = 11, height = 5, dpi = 150)
cat("Saved output/figures/fig3_model_comparison.png\n")

cat("\n── All outputs written to output/ ──\n")
cat("Tables : output/tables/ols_results.csv, ols_stargazer.html,\n")
cat("         lasso_results.csv, lasso_metrics.csv, model_comparison.csv\n")
cat("Figures: output/figures/fig1_timeseries.png, fig2_lasso_coefs.png, fig3_model_comparison.png\n")
