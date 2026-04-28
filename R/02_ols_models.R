library(tidyverse)
library(stargazer)

panel <- read_csv("data/processed/monthly_panel.csv") %>%
  mutate(date = as.Date(date))

macro_vars     <- c("michigan_inflation", "cleveland_inflation", "spf_cpi_forecast",
                    "unemployment", "ppi", "oil_wti", "pce_yoy")
market_vars    <- c("kalshi_implied_cpi")
sentiment_vars <- c("gdelt_tone", "google_trends_inflation")
all_pred_vars  <- c(macro_vars, market_vars, sentiment_vars)

panel_std <- panel %>%
  mutate(across(all_of(all_pred_vars), ~ as.numeric(scale(.))))

rmse <- function(model) sqrt(mean(residuals(model)^2))

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

results <- tibble(
  Model  = c("Macro/Surveys OLS", "Prediction Markets OLS", "Sentiment OLS"),
  N      = c(nobs(fit_macro), nobs(fit_market), nobs(fit_sentiment)),
  R2     = round(c(summary(fit_macro)$r.squared,
                    summary(fit_market)$r.squared,
                    summary(fit_sentiment)$r.squared), 3),
  Adj_R2 = round(c(summary(fit_macro)$adj.r.squared,
                    summary(fit_market)$adj.r.squared,
                    summary(fit_sentiment)$adj.r.squared), 3),
  RMSE   = round(c(rmse(fit_macro), rmse(fit_market), rmse(fit_sentiment)), 3)
)

print(results)
write_csv(results, "output/tables/ols_results.csv")
cat("OLS summary saved: output/tables/ols_results.csv\n")

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
cat("Stargazer table saved: output/tables/ols_stargazer.html\n")
