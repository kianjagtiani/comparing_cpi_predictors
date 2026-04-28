pkgs <- c("glmnet", "tidyverse", "ggplot2", "stargazer",
          "lubridate", "readr", "dplyr", "tidyr",
          "scales", "gridExtra", "kableExtra")
new <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(new)) install.packages(new, repos = "https://cloud.r-project.org")
cat("All packages ready.\n")
