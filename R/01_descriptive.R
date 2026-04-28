library(tidyverse)
library(ggplot2)
library(scales)

panel <- read_csv("data/processed/monthly_panel.csv") %>%
  mutate(date = as.Date(date))

dir.create("output/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("output/tables",  recursive = TRUE, showWarnings = FALSE)

# ── Figure 1: Time series overlay ─────────────────────────────────────
plot_data <- panel %>%
  select(date,
         `Actual CPI YoY`     = cpi_yoy,
         `Michigan Survey`    = michigan_inflation,
         `SPF Forecast`       = spf_cpi_forecast,
         `Kalshi Implied CPI` = kalshi_implied_cpi,
         `GDELT Tone (scaled)`= gdelt_tone) %>%
  mutate(`GDELT Tone (scaled)` = `GDELT Tone (scaled)` * -1) %>%
  pivot_longer(-date, names_to = "series", values_to = "value") %>%
  drop_na()

p1 <- ggplot(plot_data, aes(x = date, y = value, color = series, linetype = series)) +
  geom_line(linewidth = 0.9) +
  scale_x_date(date_breaks = "6 months", date_labels = "%b %Y") +
  scale_color_manual(values = c(
    "Actual CPI YoY"      = "black",
    "Michigan Survey"     = "#2196F3",
    "SPF Forecast"        = "#1565C0",
    "Kalshi Implied CPI"  = "#FF5722",
    "GDELT Tone (scaled)" = "#4CAF50"
  )) +
  labs(
    title    = "CPI YoY vs. Prediction Sources (Jan 2022-Apr 2026)",
    subtitle = "GDELT tone inverted and rescaled for visual comparison",
    x = NULL, y = "Percent / Index",
    color = NULL, linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("output/figures/fig1_timeseries.png", p1, width = 10, height = 5, dpi = 150)
cat("Figure 1 saved: output/figures/fig1_timeseries.png\n")

# ── Correlation table ──────────────────────────────────────────────────
predictor_cols <- c(
  "michigan_inflation", "cleveland_inflation", "spf_cpi_forecast",
  "unemployment", "ppi", "oil_wti", "pce_yoy",
  "kalshi_implied_cpi",
  "gdelt_tone", "google_trends_inflation"
)

cor_data <- panel %>% select(cpi_yoy_next, all_of(predictor_cols)) %>% drop_na()
cor_vals <- cor(cor_data)[1, -1]

cor_df <- tibble(
  Variable    = names(cor_vals),
  Category    = c(rep("Macro/Survey", 7), "Prediction Market", "Sentiment", "Sentiment"),
  Correlation = round(cor_vals, 3)
) %>% arrange(desc(abs(Correlation)))

print(cor_df, n = 20)
write_csv(cor_df, "output/tables/correlation_table.csv")
cat("Correlation table saved: output/tables/correlation_table.csv\n")
