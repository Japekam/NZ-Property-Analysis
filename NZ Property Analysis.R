# --------------------------------------------------------------------------
# Project: NZ Housing Market Analysis & Prediction (Revised & Comprehensive)
# --------------------------------------------------------------------------

# -----------------------
# Load Libraries
# -----------------------

# Data Manipulation & Core
library(plyr)      # load it BEFORE dplyr to avoid conflicts.
library(tidyverse) # Includes dplyr, ggplot2, readr, etc.
library(lubridate) # For date-time manipulation.
library(janitor)   # For cleaning column names.
library(zoo)       # For as.yearmon, rollapply, rollmean.

# Time Series Specific
library(tseries)   # For time series tests like adf.test
library(forecast)  # For ARIMA, auto.arima, forecast, checkresiduals, ndiffs

# Machine Learning & Evaluation
library(tidymodels)  # Meta-package for modern modelling: recipes, rsample, parsnip, yardstick, etc.
# library(recipes)     # For ML preprocessing (loaded by tidymodels)
# library(rsample)     # For initial_split (loaded by tidymodels)
library(caret)       # For ML model training (can also use tidymodels workflows)
library(glmnet)      # For Lasso/Elastic Net regression.
library(randomForest)# For Random Forest algorithm.
library(xgboost)     # For XGBoost algorithm.
# library(Metrics)   # For RMSE - prefer yardstick::rmse_vec from tidymodels, loaded via 'tidymodels'

# Plotting & Visualization
library(corrplot)    # For correlation matrix plots.
library(ggthemes)    # For additional ggplot themes.
library(scales)      # For formatting axes (dollar, percent, comma).
library(RColorBrewer)# For colour palettes.
library(ggpubr)      # For easily adding stats like correlation to ggplot (e.g., stat_cor).
library(ggrepel)     # For geom_text_repel to avoid overlapping text labels in plots.


# -----------------------
# Load Data
# -----------------------
rental_raw <- read_csv("rental.csv") %>% clean_names()
property_raw <- read_csv("property.csv") %>% clean_names()
dwellings_raw <- read_csv("dwellings.csv") %>% clean_names()

# ------------------------------------
# Standardise TIME_REF and Create Date Column
# ------------------------------------
# Function to convert YYYY.MM style TIME_REF to a date (1st of the month)
convert_time_ref_to_date <- function(df, time_ref_col_name = "time_ref") {
  df <- df %>%
    mutate(
      # Standardise to ensure two decimal places, e.g., 2007.1 becomes 2007.10
      time_ref_numeric = as.numeric(sprintf("%.2f", .data[[time_ref_col_name]])),
      year = floor(time_ref_numeric), # Extract year
      month_val = round((time_ref_numeric %% 1) * 100), # Extract month as 1-12
      date = make_date(year, month_val, 1) # Create a date object
    ) %>%
    select(-time_ref_numeric) # Remove intermediate column
  return(df)
}

rental <- convert_time_ref_to_date(rental_raw, "time_ref")
property <- convert_time_ref_to_date(property_raw, "time_ref")
dwellings <- convert_time_ref_to_date(dwellings_raw, "time_ref")

# -------------------------------------------------
# Map and Aggregate Property Data to Rental Regions
# -------------------------------------------------
# Define specific locations that constitute the "Rest of..." broader rental regions
north_island_rest_locations <- c("Northland", "Waikato", "Bay of Plenty", "Gisborne", "Hawke's Bay", "Taranaki", "Manawatu")
south_island_rest_locations <- c("Tasman", "Nelson", "Marlborough", "West Coast", "Otago", "Southland")

# 1. Aggregate property data for "Rest of North Island"
property_roni <- property %>%
  filter(location %in% north_island_rest_locations) %>%
  group_by(date, year, month_val, time_ref) %>% # Group by all original time columns for consistent aggregation
  summarise(median_price = mean(median_price, na.rm = TRUE), .groups = "drop") %>% # Calculate mean of median_prices
  mutate(region = "Rest of North Island")

# 2. Aggregate property data for "Rest of South Island"
property_rosi <- property %>%
  filter(location %in% south_island_rest_locations) %>%
  group_by(date, year, month_val, time_ref) %>%
  summarise(median_price = mean(median_price, na.rm = TRUE), .groups = "drop") %>%
  mutate(region = "Rest of South Island")

# 3. Prepare property data for directly matching regions and the national total
property_direct_match <- property %>%
  filter(location %in% c("Auckland", "Wellington", "Canterbury", "NZ total")) %>%
  mutate(region = case_when(
    location == "NZ total" ~ "National", # Map "NZ total" to "National" for joining with rental data
    TRUE ~ location # Keep Auckland, Wellington, Canterbury as is
  )) %>%
  select(date, year, month_val, time_ref, region, median_price) # Select necessary columns

# Combine all processed property data segments
property_mapped <- bind_rows(
  property_direct_match,
  property_roni,
  property_rosi
) %>%
  select(date, region, median_price, year, month_val, time_ref) %>% # Standardise columns
  distinct(date, region, .keep_all = TRUE) # Ensure unique entries per date/region

# -----------------------
# Merge Datasets
# -----------------------
# Join rental data with the mapped property data
merged_rp <- rental %>%
  left_join(property_mapped, by = c("date", "region"), suffix = c("_rent", "_prop")) %>%
  # Clean up duplicated or suffixed columns from the join, prioritising columns from `rental` if names clash
  select(-any_of(c("year_prop", "month_val_prop", "time_ref_prop", "time_ref_rent.y", "year.y", "month_val.y"))) %>% 
  rename(year = year_rent, month_val = month_val_rent, time_ref = time_ref_rent)


# Prepare quarterly dwellings data for joining with the monthly merged_rp data
merged_rp <- merged_rp %>%
  mutate(
    quarter_val = quarter(date), # Determine quarter for each month
    # Create a join key: the first month of the quarter for each date
    dwelling_join_month = case_when(
      quarter_val == 1 ~ 1, quarter_val == 2 ~ 4,
      quarter_val == 3 ~ 7, quarter_val == 4 ~ 10
    ),
    dwelling_date_join = make_date(year, dwelling_join_month, 1)
  )

# Select and rename columns from dwellings data for clarity before join
dwellings_for_join <- dwellings %>%
  select(date_dwelling_source = date, owner_occupied, provided_free, rented, total)

# Join with dwellings data
full_data <- merged_rp %>%
  left_join(dwellings_for_join, by = c("dwelling_date_join" = "date_dwelling_source")) %>%
  # Dwellings data is quarterly; fill values for all months within the same quarter
  group_by(year, quarter_val) %>% 
  fill(owner_occupied, provided_free, rented, total, .direction = "downup") %>% # Fill NAs within each quarter
  ungroup() %>%
  filter(!is.na(total)) # Ensure that dwellings data is present after filling

# -----------------------
# Feature Engineering
# -----------------------
full_data <- full_data %>%
  group_by(region) %>% # Group by region for lag/rolling calculations to be region-specific
  arrange(date) %>%    # Ensure data is sorted chronologically within each region
  mutate(
    # National dwelling ratios (repeated for each region, as source is national)
    rented_ratio_national = rented / total,
    owner_ratio_national = owner_occupied / total,
    free_ratio_national = provided_free / total,
    
    # Price-to-Rent CPI ratio (regional)
    price_rent_cpi_ratio = ifelse(rent_cpi > 0 & !is.na(median_price) & median_price > 0, median_price / rent_cpi, NA),
    
    # Log transformations (regional)
    log_median_price = ifelse(!is.na(median_price) & median_price > 0, log(median_price), NA),
    log_rent_cpi = ifelse(!is.na(rent_cpi) & rent_cpi > 0, log(rent_cpi), NA),
    
    # Lagged values for growth calculation (regional)
    median_price_lag = lag(median_price, 1), # Previous month's median price
    rent_cpi_lag = lag(rent_cpi, 1),         # Previous month's rent CPI
    
    # Monthly growth rates (regional)
    price_growth_monthly = ifelse(!is.na(median_price_lag) & median_price_lag != 0, (median_price - median_price_lag) / median_price_lag, NA),
    rent_growth_monthly = ifelse(!is.na(rent_cpi_lag) & rent_cpi_lag != 0, (rent_cpi - rent_cpi_lag) / rent_cpi_lag, NA),
    
    # Rolling averages for smoothing (regional)
    rent_cpi_roll_avg3 = rollmean(rent_cpi, 3, fill = NA, align = "right", na.rm = TRUE), # 3-month rolling average
    median_price_roll_avg3 = rollmean(median_price, 3, fill = NA, align = "right", na.rm = TRUE) # 3-month rolling average
  ) %>%
  ungroup() %>%
  # Remove initial rows that would have NAs due to lag/rollmean (window of 3 means first 2 rows per group are NA)
  filter(date >= (min(full_data$date[which(!is.na(full_data$owner_occupied))], na.rm=TRUE) %m+% months(2))) 

print(paste("Final full_data rows after FE and initial NA filter:", nrow(full_data)))

# ------------------------------------------------------
# Enhanced Exploratory Data Analysis (EDA)
# ------------------------------------------------------

print("Starting EDA plots with enhanced visualisations...")

# --- Define a Global Custom Theme for Consistent, Professional Look ---
# Base theme is theme_bw() for a clean look with panel borders.
# Light grey backgrounds for plot and panel.
# Improved gridlines and text elements.
theme_set(theme_bw(base_size = 12) + 
            theme(
              # Text elements
              plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0.5, colour = "grey10"),
              plot.subtitle = element_text(size = rel(1.0), hjust = 0.5, colour = "grey30"),
              plot.caption = element_text(size = rel(0.8), hjust = 0, face = "italic", colour = "grey40"),
              axis.title = element_text(face = "bold", size = rel(1.0), colour = "grey20"),
              axis.text = element_text(size = rel(0.9), colour = "grey30"),
              legend.title = element_text(face = "bold", size = rel(0.9), colour = "grey20"),
              legend.text = element_text(size = rel(0.85), colour = "grey30"),
              strip.text = element_text(face = "bold", size = rel(0.9), colour = "grey10"), # Facet labels
              
              # Backgrounds
              plot.background = element_rect(fill = "grey95", colour = NA), # Overall plot background
              panel.background = element_rect(fill = "grey90", colour = NA), # Plotting area background
              legend.background = element_rect(fill = "grey95", colour = "grey80"), # Legend background
              strip.background = element_rect(fill="grey80", colour="grey70"), # Facet strip background
              
              # Gridlines
              panel.grid.major = element_line(colour = "grey80", linewidth = 0.25),
              panel.grid.minor = element_line(colour = "grey85", linewidth = 0.15),
              
              # Legend position
              legend.position = "bottom"
            )) 

# Define key regions for focused plots consistently
key_regions_focus <- c("Auckland", "Wellington", "Canterbury", "Rest of North Island", "Rest of South Island", "National")

# 1. Rent CPI Trend by Broad Region
plot1_data <- full_data %>% filter(region %in% key_regions_focus & !is.na(rent_cpi))
if (nrow(plot1_data) > 0) {
  plot1 <- ggplot(plot1_data, aes(x = date, y = rent_cpi, color = region, group = region)) +
    geom_line(alpha = 0.9, linewidth = 1) +
    scale_colour_brewer(palette = "Dark2", name = "Region:") +
    scale_y_continuous(labels = comma) +
    labs(title = "Monthly Rent CPI Over Time by Broad Region",
         x = "Date", y = "Rent CPI") +
    guides(color = guide_legend(nrow = 1)) # Ensure legend is single row if at bottom
  print(plot1)
} else { print("Plot 1: No data after filtering.")}

# 2. Median Property Price Trend by Broad Region
plot2_data <- full_data %>% filter(region %in% key_regions_focus & !is.na(median_price))
if (nrow(plot2_data) > 0) {
  plot2 <- ggplot(plot2_data, aes(x = date, y = median_price, color = region, group = region)) +
    geom_line(alpha = 0.9, linewidth = 1) +
    scale_colour_brewer(palette = "Set1", name = "Region:") +
    scale_y_continuous(labels = dollar_format(prefix = "$", scale = 1/1000, suffix = "K")) +
    labs(title = "Monthly Median Property Price by Broad Region",
         subtitle = "'Rest of...' regions use averaged prices. 'National' uses 'NZ total'.",
         x = "Date", y = "Median Property Price") +
    guides(color = guide_legend(nrow = 1))
  print(plot2)
} else { print("Plot 2: No data after filtering.")}

# 3. Correlation Matrix
# (corrplot is not ggplot2 based, so theme_set doesn't apply. It has its own styling.)
numeric_cols_for_corr <- c("rent_cpi", "median_price", "rented_ratio_national", "owner_ratio_national", 
                           "price_rent_cpi_ratio", "price_growth_monthly", "rent_growth_monthly")
correlation_data <- full_data %>% select(all_of(numeric_cols_for_corr)) %>% drop_na()
if(nrow(correlation_data) > 1 && ncol(correlation_data) > 1) {
  cor_matrix <- cor(correlation_data)
  # Using a light background for the corrplot itself via its own parameters if possible,
  # or just accepting its default which is usually clear.
  # For corrplot, bg parameter can change background of the plot area.
  corrplot(cor_matrix, method = "color", type = "upper", order = "hclust",
           addCoef.col = "black", tl.col = "black", tl.srt = 45, tl.cex = 0.8,
           number.cex = 0.7, diag = FALSE, col = brewer.pal(n=10, name="RdYlBu"),
           bg = "grey95") # Added background colour to corrplot
  title("Correlation Matrix of Key Housing Variables", line = 3, cex.main=1.1, col.main="grey10")
} else { print("Plot 3: Not enough data for correlation plot.") }

# 4. Distribution of Rent CPI - Faceted by Broad Region
plot4_data <- full_data %>% filter(region %in% key_regions_focus & !is.na(rent_cpi))
if(nrow(plot4_data) > 0){
  plot4 <- ggplot(plot4_data, aes(x = rent_cpi)) +
    geom_histogram(aes(y = after_stat(density), fill = region), alpha = 0.8, bins = 25, show.legend = FALSE) +
    geom_density(aes(color = region), linewidth = 0.9, show.legend = FALSE) +
    scale_fill_brewer(palette = "Pastel1") + scale_color_brewer(palette = "Set1") +
    facet_wrap(~region, scales = "free") +
    labs(title = "Distribution of Rent CPI by Broad Region", x = "Rent CPI", y = "Density") +
    theme(strip.text = element_text(size=rel(0.85))) # Adjust facet label size relative to base
  print(plot4)
} else { print("Plot 4: No data after filtering.")}

# 5. Distribution of Median Property Prices - Faceted by Broad Region
plot5_data <- full_data %>% filter(region %in% key_regions_focus & !is.na(median_price))
if(nrow(plot5_data) > 0){
  plot5 <- ggplot(plot5_data, aes(x = median_price)) +
    geom_histogram(aes(y = after_stat(density), fill = region), alpha = 0.8, bins = 25, show.legend = FALSE) +
    geom_density(aes(color = region), linewidth = 0.9, show.legend = FALSE) +
    scale_fill_brewer(palette = "Pastel2") + scale_color_brewer(palette = "Set2") +
    scale_x_continuous(labels = dollar_format(prefix="$", scale=1/1000, suffix="K")) +
    facet_wrap(~region, scales = "free") +
    labs(title = "Distribution of Median Prices by Broad Region", x = "Median Price", y = "Density") +
    theme(strip.text = element_text(size=rel(0.85)))
  print(plot5)
} else { print("Plot 5: No data after filtering.")}

# 6. Rent CPI vs Median Price - Faceted by Broad Region
plot6_data <- full_data %>% filter(region %in% key_regions_focus & !is.na(median_price) & !is.na(rent_cpi))
if(nrow(plot6_data) > 0){
  plot6 <- ggplot(plot6_data, aes(x = median_price, y = rent_cpi)) +
    geom_point(aes(color = region), alpha = 0.4, show.legend = FALSE, size=1) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth=0.6) +
    scale_x_continuous(labels = dollar_format(prefix="$", scale=1/1000, suffix="K")) +
    scale_color_brewer(palette = "Dark2") +
    facet_wrap(~region, scales = "free") + # "free" scales allow each facet to have its own axis range
    labs(title = "Rent CPI vs. Median Property Price", 
         subtitle = "Relationship by Broad Region", x = "Median Property Price", y = "Rent CPI") +
    theme(strip.text = element_text(size=rel(0.85))) +
    ggpubr::stat_cor(aes(label = ..r.label..), method = "pearson", label.x.npc = 0.05, label.y.npc = 0.95, size=3)
  print(plot6)
} else { print("Plot 6: No data after filtering.")}

# 7. National Dwelling Composition Over Time - Stacked Area
national_dwelling_ratios_long <- full_data %>%
  filter(!is.na(rented_ratio_national) & !is.na(owner_ratio_national) & !is.na(free_ratio_national)) %>%
  distinct(date, .keep_all = TRUE) %>% 
  select(date, `Rented` = rented_ratio_national, `Owner Occupied` = owner_ratio_national, `Provided Free` = free_ratio_national) %>%
  pivot_longer(cols = -date, names_to = "dwelling_type", values_to = "ratio")

if(nrow(national_dwelling_ratios_long) > 0){
  plot7 <- ggplot(national_dwelling_ratios_long, aes(x = date, y = ratio, fill = dwelling_type)) +
    geom_area(alpha = 0.85, position = "stack") + # Use "stack" for proportions
    scale_y_continuous(labels = percent_format()) +
    scale_fill_brewer(palette = "Accent", name = "Dwelling Type:") +
    labs(title = "National Dwelling Composition Over Time", x = "Date", y = "Proportion of Total Dwellings")
  print(plot7)
} else { print("Plot 7: No data after filtering.")}

# 8. Property Price Growth (Monthly) - Boxplot by Year, Faceted by Broad Region
plot8_data <- full_data %>%
  filter(region %in% key_regions_focus & !is.na(price_growth_monthly) & is.finite(price_growth_monthly)) %>%
  mutate(year_factor = factor(year)) # Ensure year is treated as a factor for discrete x-axis
if(nrow(plot8_data) > 0){
  plot8 <- ggplot(plot8_data, aes(x = year_factor, y = price_growth_monthly, fill = region)) +
    geom_boxplot(show.legend = FALSE, outlier.alpha = 0.3, outlier.size=0.8, linewidth=0.3) + 
    geom_hline(yintercept = 0, linetype = "dotted", color = "red4", linewidth=0.6) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
    scale_fill_brewer(palette = "Pastel2") +
    facet_wrap(~region, scales = "free_y", ncol = 3) +
    labs(title = "Distribution of Monthly Property Price Growth by Year", 
         subtitle="Faceted by Broad Region", x = "Year", y = "Monthly Price Growth") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size=rel(0.8)), 
          strip.text = element_text(size=rel(0.85)))
  print(plot8)
} else { print("Plot 8: No data after filtering.")}

# 9. Rent CPI Growth (Monthly) - Line plot for Broad Regions
plot9_data <- full_data %>% filter(region %in% key_regions_focus & !is.na(rent_growth_monthly) & is.finite(rent_growth_monthly))
if(nrow(plot9_data) > 0){
  plot9 <- ggplot(plot9_data, aes(x = date, y = rent_growth_monthly, color = region, group = region)) +
    geom_line(alpha = 0.8, linewidth=0.9) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey30", linewidth=0.6) +
    scale_y_continuous(labels = percent_format(accuracy = 0.1)) +
    scale_color_brewer(palette = "Set2", name="Region:") +
    labs(title = "Monthly Rent CPI Growth Over Time", 
         subtitle="Faceted by Broad Region", x = "Date", y = "Monthly Rent CPI Growth") +
    facet_wrap(~region, scales="free_y", ncol=2) + # Faceting can be clearer than many lines
    theme(legend.position = "none", strip.text = element_text(size=rel(0.85))) # Remove legend if faceting
  print(plot9)
} else { print("Plot 9: No data after filtering.")}

# Plot 10: Price-to-Rent CPI Ratio Heatmap by Region and Year
heatmap_data_price_rent_ratio <- full_data %>%
  filter(!is.na(price_rent_cpi_ratio) & is.finite(price_rent_cpi_ratio) & region != "National") %>%
  group_by(year, region) %>% summarise(avg_price_rent_cpi_ratio = mean(price_rent_cpi_ratio, na.rm = TRUE), .groups = "drop")
if(nrow(heatmap_data_price_rent_ratio) > 0){
  plot10 <- ggplot(heatmap_data_price_rent_ratio, 
                   aes(x = factor(year), y = fct_reorder(region, avg_price_rent_cpi_ratio, .fun = median, .desc = TRUE, na.rm=TRUE), 
                       fill = avg_price_rent_cpi_ratio)) +
    geom_tile(color = "white", linewidth=0.2) + 
    scale_fill_viridis_c(option = "plasma", name = "Avg. Price/\nRent CPI Ratio", labels = comma, direction = -1) +
    labs(title = "Heatmap: Average Price-to-Rent CPI Ratio", 
         subtitle="Excluding 'National'. 'Rest of...' use averaged prices.", x = "Year", y = "Region") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size=rel(0.9)), 
          legend.position = "right", 
          panel.grid = element_blank(), panel.border = element_blank()) # Cleaner heatmap
  print(plot10)
} else { print("Plot 10: No data after filtering.")}

# Plot 11: Smoothed Rent CPI (3-month Rolling Avg) - Faceted by Broad region
plot11_data <- full_data %>%
  filter(region %in% key_regions_focus & !is.na(rent_cpi_roll_avg3) & !is.na(rent_cpi))
if(nrow(plot11_data) > 0){
  plot11 <- ggplot(plot11_data, aes(x = date)) +
    geom_line(aes(y = rent_cpi, colour = "Actual CPI"), alpha = 0.5, linewidth=0.7) +
    geom_line(aes(y = rent_cpi_roll_avg3, colour = "3-Month Rolling Avg"), linewidth = 1.1) +
    facet_wrap(~region, scales = "free_y", ncol=3) +
    scale_colour_manual(values = c("Actual CPI" = "grey60", "3-Month Rolling Avg" = "steelblue3"), name="Metric:") +
    labs(title = "Smoothed Rent CPI (3-Month Rolling Average)", 
         subtitle="Faceted by Broad Region", x = "Date", y = "Rent CPI") +
    theme(legend.position = "top", strip.text = element_text(size=rel(0.85)))
  print(plot11)
} else { print("Plot 11: No data after filtering.")}

# Plot 12: Volatility of Property Price Growth (12-month Rolling SD) by Broad Region
plot12_data <- full_data %>%
  group_by(region) %>% arrange(date) %>%
  mutate(price_growth_volatility_12m = zoo::rollapply(price_growth_monthly, width = 12, FUN = sd, fill = NA, align = "right", na.rm = TRUE)) %>%
  ungroup() %>%
  filter(region %in% key_regions_focus & !is.na(price_growth_volatility_12m) & is.finite(price_growth_volatility_12m))
if(nrow(plot12_data) > 0){
  plot12 <- ggplot(plot12_data, aes(x = date, y = price_growth_volatility_12m, color = region, group=region)) +
    geom_line(linewidth = 0.9) +
    scale_y_continuous(labels = percent_format(accuracy=0.1)) +
    scale_color_brewer(palette = "Set1", name="Region:") +
    labs(title = "Monthly Property Price Growth Volatility",
         subtitle="12-Month Rolling Standard Deviation, by Broad Region",
         x = "Date", y = "Std. Dev of Monthly Price Growth") +
    facet_wrap(~region, scales="free_y", ncol=2) + # Faceting might be clearer
    theme(legend.position = "none", strip.text = element_text(size=rel(0.85))) 
  print(plot12)
} else { print("Plot 12: No data after filtering.")}

# Plot 13: Top 5 Regions by Current Average Price-to-Rent CPI Ratio (Excluding National)
current_year_val <- max(full_data$year, na.rm = TRUE)
top5_price_rent_regions <- full_data %>%
  filter(year == current_year_val & !is.na(price_rent_cpi_ratio) & is.finite(price_rent_cpi_ratio) & region != "National") %>%
  group_by(region) %>%
  summarise(avg_ratio = mean(price_rent_cpi_ratio, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(avg_ratio)) %>%
  slice_head(n = 5)
if(nrow(top5_price_rent_regions) > 0){
  plot13 <- ggplot(top5_price_rent_regions, aes(x = reorder(region, avg_ratio), y = avg_ratio, fill = region)) +
    geom_col(show.legend = FALSE, alpha=0.85) +
    geom_text(aes(label=sprintf("%.1f", avg_ratio)), hjust = -0.2, size=3.5, fontface="bold", color="grey20") +
    coord_flip(ylim=c(0, max(top5_price_rent_regions$avg_ratio, na.rm=T)*1.18)) + 
    scale_fill_brewer(palette = "Pastel1") +
    labs(title = paste("Top 5 Regions by Avg. Price-to-Rent CPI Ratio in", current_year_val),
         subtitle = "Excludes 'National'. 'Rest of...' use averaged prices.",
         x = "Region", y = "Average Price-to-Rent CPI Ratio") +
    theme(panel.grid.major.x = element_line(colour = "grey80"), # Keep horizontal grid lines for bar chart
          panel.grid.major.y = element_blank()) # Remove vertical for coord_flip
  print(plot13)
} else { print("Plot 13: No data after filtering.")}

# Plot 14: Property Price vs. National Owner-Occupied Ratio - Faceted by Broad Region
plot14_data <- full_data %>%
  filter(region %in% key_regions_focus & !is.na(owner_ratio_national) & !is.na(median_price) & region != "National")
if(nrow(plot14_data) > 0){
  plot14 <- ggplot(plot14_data, aes(x = owner_ratio_national, y = median_price)) +
    geom_point(aes(color = region), alpha = 0.25, show.legend = FALSE, size=1) + 
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed", linewidth=0.6) +
    scale_x_continuous(labels = percent_format()) +
    scale_y_continuous(labels = dollar_format(prefix="$", scale=1/1000, suffix="K")) +
    scale_color_brewer(palette = "Set2") +
    facet_wrap(~region, scales = "free", ncol = 3) + 
    labs(title = "Median Property Price vs. National Owner-Occupied Ratio",
         subtitle = "Faceted by Broad Region (excl. National). 'Rest of...' use averaged prices.",
         x = "National % Owner-Occupied Dwellings", y = "Median Property Price") +
    theme(strip.text = element_text(size = rel(0.85)))
  print(plot14)
} else { print("Plot 14: No data after filtering.")}

# Plot 15: Regional Price & Rent CPI Growth Comparison (Most Recent Full Year)
recent_full_year_val <- if(length(unique(full_data$year[!is.na(full_data$year)])) > 1) { 
  max(full_data$year, na.rm=TRUE) -1 
} else { 
  max(full_data$year, na.rm=TRUE) 
}
growth_comparison_data <- full_data %>%
  filter(year == recent_full_year_val & 
           !is.na(price_growth_monthly) & is.finite(price_growth_monthly) &
           !is.na(rent_growth_monthly) & is.finite(rent_growth_monthly) &
           region != "National") %>% 
  group_by(region) %>%
  summarise(avg_annual_price_growth = mean(price_growth_monthly*12, na.rm = TRUE), 
            avg_annual_rent_growth = mean(rent_growth_monthly*12, na.rm = TRUE),   
            .groups = "drop") %>%
  drop_na()
if(nrow(growth_comparison_data) > 0){
  plot15 <- ggplot(growth_comparison_data, aes(x = avg_annual_rent_growth, y = avg_annual_price_growth)) +
    geom_point(aes(fill = region, size = abs(avg_annual_price_growth - avg_annual_rent_growth)), 
               alpha = 0.7, shape=21, stroke=0.5, color="white") + # Use fill for points
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "grey30", linewidth=0.7) +
    ggrepel::geom_text_repel(aes(label = region), size = 3.5, max.overlaps = 15, 
                             segment.alpha = 0.5, color="grey10", fontface="italic") + 
    scale_x_continuous(labels = percent_format()) +
    scale_y_continuous(labels = percent_format()) +
    scale_fill_brewer(palette = "Dark2", name="Region:") +
    scale_size_continuous(name="Growth Difference (Abs):") +
    labs(title = paste("Annualised Property Price Growth vs. Rent CPI Growth in", recent_full_year_val),
         subtitle="Excludes 'National'. 'Rest of...' use averaged prices.",
         x = "Average Annualised Monthly Rent CPI Growth", y = "Average Annualised Monthly Property Price Growth") +
    theme(legend.box = "horizontal")
  print(plot15)
} else { print("Plot 15: No data after filtering.")}

print("EDA plots finished.")


# --------------------------------------------------
# Machine Learning: Data Prep + Modelling + Evaluation (with Hyperparameter Tuning)
# --------------------------------------------------

# --- 1. Define Feature Set for Modelling ---
ml_feature_cols <- c(
  "median_price", "rented_ratio_national", "owner_ratio_national", "free_ratio_national",
  "price_rent_cpi_ratio", "log_median_price", "price_growth_monthly", "rent_growth_monthly"
)

# --- 2. Initial Data Preparation for ML ---
ml_data_pre_split <- full_data %>%
  select(rent_cpi, date, region, all_of(ml_feature_cols)) %>%
  drop_na(rent_cpi) 

# --- 3. Pre-ML Data Sanity Checks & Further NA Handling ---
if (nrow(ml_data_pre_split) > 0) {
  print("--- ML Data Sanity Checks ---")
  print(paste("Rows before dropping NAs in predictors:", nrow(ml_data_pre_split)))
  ml_data_final <- ml_data_pre_split %>%
    drop_na(any_of(c("median_price", "price_rent_cpi_ratio", "rented_ratio_national", "price_growth_monthly", "rent_growth_monthly")))
  print(paste("Rows after dropping NAs in key predictors:", nrow(ml_data_final)))
  if (nrow(ml_data_final) > 0 && var(ml_data_final$rent_cpi, na.rm = TRUE) < 1e-6) {
    print("Warning: Target 'rent_cpi' has near-zero variance after filtering. ML may not be meaningful.")
    ml_data_final <- data.frame() 
  }
} else {
  ml_data_final <- data.frame() 
}

# --- 4. Add Date-Derived Features Explicitly ---
if (nrow(ml_data_final) > 0) {
  ml_data_final <- ml_data_final %>%
    mutate(
      year = lubridate::year(date), month = lubridate::month(date), 
      quarter = lubridate::quarter(date), half_year = ifelse(month %in% 1:6, 1, 2) 
    )
}

# --- 5. Proceed to ML if Data is Sufficient ---
if (nrow(ml_data_final) > 100) { 
  set.seed(123) 
  split <- initial_split(ml_data_final, prop = 0.8, strata = rent_cpi)
  train_data <- training(split)
  test_data <- testing(split)
  
  print(paste("Training data rows:", nrow(train_data)))
  print(paste("Test data rows:", nrow(test_data)))
  
  rec <- recipe(rent_cpi ~ ., data = train_data) %>%
    update_role(region, new_role = "ID") %>% 
    update_role(date, new_role = "ID") %>%   
    step_mutate( 
      month = factor(month, levels = 1:12, labels = month.abb, ordered = FALSE),
      quarter = factor(quarter, levels = 1:4, labels = paste0("Q", 1:4), ordered = FALSE),
      half_year = factor(half_year, levels = 1:2, labels = paste0("H", 1:2), ordered = FALSE)
    ) %>%
    step_novel(all_nominal_predictors(), all_factor_predictors(), -all_outcomes(), -has_role("ID")) %>% 
    step_zv(all_predictors()) %>% 
    step_impute_median(all_numeric_predictors()) %>% 
    step_dummy(all_nominal_predictors(), -all_outcomes(), -has_role("ID"), one_hot = FALSE) %>% 
    step_normalize(all_numeric_predictors()) %>%
    step_corr(all_numeric_predictors(), threshold = 0.9)
  
  print("Recipe defined. Starting model training...")
  prepped_recipe_for_models <- prep(rec, training = train_data) # Prep recipe once for all models
  
  ctrl <- trainControl(method = "cv", number = 10, savePredictions = "final", allowParallel = TRUE)
  
  print("Training Linear Model...")
  model_lm <- train(prepped_recipe_for_models, data = train_data, method = "lm", trControl = ctrl, metric = "RMSE")
  print(summary(model_lm$finalModel)) 
  
  print("Training Lasso Model...")
  lasso_grid <- expand.grid(alpha = 1, lambda = 10^seq(-4, -1, length.out = 10)) 
  model_lasso <- train(prepped_recipe_for_models, data = train_data, method = "glmnet", tuneGrid = lasso_grid, trControl = ctrl, metric = "RMSE")
  print(paste("Lasso best lambda:", model_lasso$bestTune$lambda))
  
  print("Training Random Forest Model...")
  model_rf <- train(prepped_recipe_for_models, data = train_data, method = "rf", ntree = 50, tuneLength = 6, trControl = ctrl, metric = "RMSE", importance = TRUE) 
  print(paste("Random Forest best mtry:", model_rf$bestTune$mtry))
  
  print("Training XGBoost Model...")
  xgb_grid <- expand.grid(nrounds = c(100, 300, 500), max_depth = c(3, 6, 9), eta = c(0.01, 0.1, 0.3), gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8) 
  model_xgb <- train(prepped_recipe_for_models, data = train_data, method = "xgbTree", tuneGrid = xgb_grid, trControl = ctrl, metric = "RMSE", verbosity = 0) 
  print("XGBoost best params:"); print(model_xgb$bestTune) 
  
  model_list_for_resamples <- list(LM = model_lm, Lasso = model_lasso, RF = model_rf, XGB = model_xgb)
  results <- resamples(model_list_for_resamples)
  print("--- Resamples Summary (Cross-Validation Metrics) ---"); print(summary(results))
  # Check for any NA values in the resampled metrics
  if (!is.null(results) && !is.null(results$values) && nrow(results$values) > 0 && !any(is.na(results$values))) {
    
    print("--- Generating Model Comparison Dot Plots ---")
    
    # For RMSE dotplot
    tryCatch({
      # Remove explicit xlab and ylab; let dotplot create them by default.
      # Keep the 'main' argument for the title.
      plot_cv_rmse <- dotplot(results, 
                              metric = "RMSE", 
                              main = list(label = "Model Comparison (Cross-Validation RMSE)", cex = 1.1, col = "grey10")
      )
      if (!is.null(plot_cv_rmse)) {
        print(plot_cv_rmse)
      } else {
        print("RMSE dotplot object was NULL or failed to generate.")
      }
    }, error = function(e_rmse_plot) {
      print(paste("Error generating RMSE dotplot:", e_rmse_plot$message))
      # Suggest to the user to try the simplest form if it fails
      print("As a test, you could try running just: dotplot(results, metric='RMSE') in your console.")
    })
    
    # For Rsquared dotplot
    tryCatch({
      plot_cv_rsquared <- dotplot(results, 
                                  metric = "Rsquared", 
                                  main = list(label = "Model Comparison (Cross-Validation R-squared)", cex = 1.1, col = "grey10")
      )
      if (!is.null(plot_cv_rsquared)) {
        print(plot_cv_rsquared)
      } else {
        print("Rsquared dotplot object was NULL or failed to generate.")
      }
    }, error = function(e_rsq_plot) {
      print(paste("Error generating Rsquared dotplot:", e_rsq_plot$message))
      print("As a test, you could try running just: dotplot(results, metric='Rsquared') in your console.")
    })
    
  } else {
    print("No valid metric values available in resamples object for plotting, or 'results' object is NULL.")
    if(!is.null(results) && !is.null(results$values) && nrow(results$values) > 0 && any(is.na(results$values))){
      print("NA/NaN values were detected in resamples metrics earlier. This might be why plots cannot be generated.")
      print("ACTION: Run warnings() in your R console to investigate specific warnings generated during training.")
    }
  }
  
  print("--- RMSE on Test Set ---")
  rmse_values <- sapply(model_list_for_resamples, function(mod) { predict(mod, newdata = test_data) %>% yardstick::rmse_vec(truth = test_data$rent_cpi, estimate = .) })
  print(rmse_values) 
  
  if (exists("model_rf") && !is.null(model_rf)) { 
    print("Plotting RF Actual vs Predicted...")
    final_preds_rf <- predict(model_rf, newdata = test_data)
    plot_df_rf <- data.frame(Actual = test_data$rent_cpi, Predicted = as.numeric(final_preds_rf), Date = test_data$date, Region = test_data$region)
    pred_vs_actual_plot_rf <- ggplot(plot_df_rf, aes(x = Actual, y = Predicted)) + geom_point(alpha = 0.5, colour = "forestgreen") + geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black", linewidth = 0.8) + labs(title = "Actual vs. Predicted Rent CPI (Random Forest - Test Set)", x = "Actual Rent CPI", y = "Predicted Rent CPI") + coord_fixed() 
    print(pred_vs_actual_plot_rf)
  }
} else {
  print("Not enough data for ML modelling.")
}


# ==========================================================================
# FORECASTING SECTION
# ==========================================================================
# This section contains two methods for forecasting Rent CPI for a specific region.
# Method 1: Direct ARIMA forecasting on Rent CPI.
# Method 2: Recursive forecasting using the trained Random Forest model,
#           with Median Price (a key predictor) itself forecasted using ARIMA.
# ==========================================================================

# --- General Setup for Forecasting ---
# I'll choose a region to focus on for these examples.
forecast_target_region <- "Auckland" 
# Define how far out I want to forecast.
last_historical_date <- max(full_data$date, na.rm = TRUE)
forecast_horizon_start_date <- last_historical_date %m+% months(1)
forecast_horizon_end_date <- make_date(2030, 12, 1)
future_dates_for_forecast <- seq(from = forecast_horizon_start_date, to = forecast_horizon_end_date, by = "month")
num_forecast_periods <- length(future_dates_for_forecast)

print(paste("--- Starting Forecasting for Region:", forecast_target_region, "for", num_forecast_periods, "months ---"))

# Need the prepped recipe from the ML section to transform future data for RF model
if (!exists("prepped_recipe_for_models") || !"recipe" %in% class(prepped_recipe_for_models) || is.null(prepped_recipe_for_models$tr_info)) {
  if(exists("rec") && exists("train_data")){
    print("Prepping the main recipe on training data for use in forecasting...")
    prepped_recipe_for_models <- prep(rec, training = train_data)
  } else {
    stop("The prepped recipe ('prepped_recipe_for_models' or 'rec' + 'train_data') is not available. Cannot proceed with RF forecasting.")
  }
}


# ==========================================================================
# REGIONAL FORECASTING SECTION
# ==========================================================================
# This section applies two forecasting methods for Rent CPI to multiple regions.
# Method 1: Direct ARIMA forecasting on Regional Rent CPI.
# Method 2: Recursive forecasting using the trained Random Forest model,
#           with Regional Median Price (a key predictor) itself forecasted using ARIMA.
# ==========================================================================

print("--- Initialising Regional Forecasting ---")

# --- 1. General Setup for Forecasting ---
# Define how far out I want to forecast.
last_historical_date <- max(full_data$date, na.rm = TRUE)
forecast_horizon_start_date <- last_historical_date %m+% months(1)
forecast_horizon_end_date <- make_date(2030, 12, 1) # Aiming for end of 2030
future_dates_for_forecast <- seq(from = forecast_horizon_start_date, to = forecast_horizon_end_date, by = "month")
num_forecast_periods <- length(future_dates_for_forecast)

# Get the list of unique regions to iterate over
# Excluding "National" for recursive RF if its median price forecast behaves differently or is not an average.
# Or include it if you have a clear way to project its median_price. For now, let's try all.
unique_regions_to_forecast <- unique(full_data$region)
# You might want to select a subset for faster testing:
# unique_regions_to_forecast <- c("Auckland", "Wellington", "National") 

print(paste("Will attempt forecasting for the following regions:", paste(unique_regions_to_forecast, collapse=", ")))
print(paste("Forecasting for", num_forecast_periods, "months, from", forecast_horizon_start_date, "to", forecast_horizon_end_date))

# Need the prepped recipe from the ML section for the Random Forest model
if (!exists("prepped_recipe_for_models") || !"recipe" %in% class(prepped_recipe_for_models) || is.null(prepped_recipe_for_models$tr_info)) {
  if(exists("rec") && exists("train_data")){ # Assuming 'rec' is your main recipe object
    print("Prepping the main recipe on training data for use in RF forecasting...")
    prepped_recipe_for_models <- prep(rec, training = train_data)
  } else {
    stop("The prepped recipe ('prepped_recipe_for_models' or 'rec' + 'train_data') is not available. Cannot proceed with RF forecasting.")
  }
}

# Ensure my best Random Forest model is available
if (!exists("model_rf") || !("train" %in% class(model_rf))) {
  stop("My Random Forest model 'model_rf' isn't here. Need to run the ML training part first.")
}
active_forecasting_model_rf <- model_rf # This is the RF model I'll use

# Prepare last known national dwelling ratios (held constant for all regions' forecasts)
last_national_ratios_fc <- full_data %>%
  arrange(desc(date)) %>%
  distinct(rented_ratio_national, owner_ratio_national, free_ratio_national) %>%
  dplyr::slice(1) %>% # Use dplyr::slice here
  select(rented_ratio_national, owner_ratio_national, free_ratio_national)

if(nrow(last_national_ratios_fc) == 0 || any(is.na(last_national_ratios_fc))){
  print("Warning: Couldn't get complete last national ratios for forecasting. Using placeholder values.")
  last_national_ratios_fc <- data.frame(rented_ratio_national=0.3, owner_ratio_national=0.6, free_ratio_national=0.1)
}

# --- 2. Initialise Lists to Store All Forecasts ---
all_regional_arima_cpi_forecasts <- list()
all_regional_rf_cpi_forecasts <- list()

# --- 3. Loop Through Each Region for Forecasting ---
for (current_forecast_region in unique_regions_to_forecast) {
  
  print(paste0("\n === Processing Forecasts for Region: ", current_forecast_region, " ==="))
  
  # ========================================================================
  # Method 1: Direct ARIMA Forecasting for current_forecast_region Rent CPI
  # ========================================================================
  print(paste("--- Method 1: Direct ARIMA Forecast for Rent CPI in", current_forecast_region, "---"))
  
  regional_rent_cpi_history <- full_data %>%
    filter(region == current_forecast_region & !is.na(rent_cpi)) %>%
    select(date, rent_cpi) %>%
    arrange(date)
  
  regional_arima_cpi_forecast_obj_loop <- NULL # Initialise for this region
  
  if(nrow(regional_rent_cpi_history) < 36) { 
    print(paste("Not enough historical Rent CPI data for ARIMA modelling for region:", current_forecast_region))
  } else {
    start_year_arima_cpi <- year(min(regional_rent_cpi_history$date))
    start_month_arima_cpi <- month(min(regional_rent_cpi_history$date))
    ts_regional_cpi_direct <- ts(regional_rent_cpi_history$rent_cpi, 
                                 start = c(start_year_arima_cpi, start_month_arima_cpi), 
                                 frequency = 12) 
    
    # print(autoplot(ts_regional_cpi_direct) + ggtitle(paste("Monthly Rent CPI -", current_forecast_region))) # Optional: plot input series
    
    regional_arima_cpi_model_loop <- NULL
    tryCatch({
      regional_arima_cpi_model_loop <- auto.arima(ts_regional_cpi_direct, 
                                                  seasonal = TRUE, stepwise = FALSE, approximation = FALSE,
                                                  allowdrift = TRUE, lambda = "auto")
      print(paste("ARIMA model summary for Rent CPI in", current_forecast_region, ":", regional_arima_cpi_model_loop$method))
      # print(checkresiduals(regional_arima_cpi_model_loop)) # Optional: view residuals for each
      
      regional_arima_cpi_forecast_obj_loop <- forecast(regional_arima_cpi_model_loop, h = num_forecast_periods)
      
      # Store the forecast
      all_regional_arima_cpi_forecasts[[current_forecast_region]] <- tibble(
        date = future_dates_for_forecast,
        region = current_forecast_region,
        rent_cpi_forecast_arima = as.numeric(regional_arima_cpi_forecast_obj_loop$mean),
        lower80_arima = as.numeric(regional_arima_cpi_forecast_obj_loop$lower[,1]),
        upper80_arima = as.numeric(regional_arima_cpi_forecast_obj_loop$upper[,1]),
        lower95_arima = as.numeric(regional_arima_cpi_forecast_obj_loop$lower[,2]),
        upper95_arima = as.numeric(regional_arima_cpi_forecast_obj_loop$upper[,2])
      )
      
      # Plotting forecast for current region (ARIMA method)
      current_arima_plot_data <- bind_rows(
        regional_rent_cpi_history %>% rename(value=rent_cpi) %>% mutate(type="Actual"),
        all_regional_arima_cpi_forecasts[[current_forecast_region]] %>% rename(value=rent_cpi_forecast_arima) %>% mutate(type="Forecast (ARIMA)")
      )
      arima_plot <- ggplot(current_arima_plot_data, aes(x=date, y=value, color=type)) + geom_line(linewidth=0.8) +
        geom_ribbon(data=filter(all_regional_arima_cpi_forecasts[[current_forecast_region]], !is.na(lower95_arima)), 
                    aes(ymin=lower95_arima, ymax=upper95_arima, x=date), fill="skyblue", alpha=0.3, inherit.aes=FALSE) +
        geom_ribbon(data=filter(all_regional_arima_cpi_forecasts[[current_forecast_region]], !is.na(lower80_arima)), 
                    aes(ymin=lower80_arima, ymax=upper80_arima, x=date), fill="skyblue", alpha=0.5, inherit.aes=FALSE) +
        scale_colour_manual(values = c("Actual" = "black", "Forecast (ARIMA)" = "deepskyblue3")) +
        labs(title=paste("Rent CPI ARIMA Forecast:", current_forecast_region), subtitle=regional_arima_cpi_model_loop$method, y="Rent CPI")
      print(arima_plot)
      
    }, error = function(e_arima_cpi) {
      print(paste("Direct ARIMA for Rent CPI failed for", current_forecast_region, ":", e_arima_cpi$message))
    })
  } 
  
  # ========================================================================
  # Method 2: Recursive RF Forecast for Rent CPI in current_forecast_region
  #           (using ARIMA-forecasted Median Price)
  # ========================================================================
  print(paste("--- Method 2: Recursive RF Forecast for Rent CPI in", current_forecast_region, "---"))
  
  # --- 2a. Forecast 'median_price' for current_forecast_region using ARIMA ---
  historical_median_price_data_rf <- full_data %>%
    filter(region == current_forecast_region & !is.na(median_price)) %>%
    select(date, median_price) %>%
    arrange(date)
  
  future_median_prices_for_rf_loop <- NULL 
  
  if(nrow(historical_median_price_data_rf) < 24) { 
    print(paste("Warning: Not enough historical median_price for", current_forecast_region, "to fit ARIMA. Holding last price constant."))
    if(nrow(historical_median_price_data_rf) > 0) {
      future_median_prices_for_rf_loop <- rep(tail(historical_median_price_data_rf$median_price, 1), num_forecast_periods)
    } else {
      future_median_prices_for_rf_loop <- rep(NA, num_forecast_periods) 
    }
  } else {
    median_price_ts_rf <- ts(historical_median_price_data_rf$median_price,
                             start = c(year(min(historical_median_price_data_rf$date)), month(min(historical_median_price_data_rf$date))),
                             frequency = 12)
    
    arima_median_price_model_rf_loop <- NULL
    tryCatch({
      arima_median_price_model_rf_loop <- auto.arima(median_price_ts_rf, seasonal = TRUE, stepwise = TRUE, 
                                                     approximation = FALSE, allowdrift = TRUE, lambda = 0) # Using log transform
      future_median_prices_forecast_obj_rf_loop <- forecast(arima_median_price_model_rf_loop, h = num_forecast_periods)
      future_median_prices_for_rf_loop <- as.numeric(future_median_prices_forecast_obj_rf_loop$mean)
      # Optional: Plot helper forecast for median price
      # print(autoplot(future_median_prices_forecast_obj_rf_loop) + ggtitle(paste("Helper: Median Price ARIMA Forecast for", current_forecast_region)))
    }, error = function(e_arima_mp) {
      print(paste("ARIMA for median_price input failed for", current_forecast_region, ":", e_arima_mp$message))
      if(nrow(historical_median_price_data_rf) > 0) {
        future_median_prices_for_rf_loop <<- rep(tail(historical_median_price_data_rf$median_price, 1), num_forecast_periods)
      } else {
        future_median_prices_for_rf_loop <<- rep(NA, num_forecast_periods)
      }
    })
  }
  
  # If future_median_prices_for_rf_loop is still NULL or all NA, we can't proceed with RF forecast for this region
  if(is.null(future_median_prices_for_rf_loop) || all(is.na(future_median_prices_for_rf_loop))) {
    print(paste("Cannot proceed with RF forecast for", current_forecast_region, "due to missing future median prices."))
  } else {
    # --- 2b. Prepare Last Known Actual Data for current_forecast_region ---
    last_observation_rf_startup <- full_data %>%
      filter(region == current_forecast_region) %>%
      filter(date == max(date, na.rm = TRUE)) %>%
      select(any_of(colnames(ml_data_final)), rent_cpi_lag, median_price_lag) %>% 
      mutate(year = lubridate::year(date), month = lubridate::month(date), quarter = lubridate::quarter(date), half_year = ifelse(month %in% 1:6, 1, 2))
    
    if(nrow(last_observation_rf_startup) == 0) {
      print(paste("No final historical data for RF for", current_forecast_region))
    } else {
      rent_cpi_current_is_previous_for_next <- last_observation_rf_startup$rent_cpi 
      rent_cpi_previous_is_two_ago_for_next <- last_observation_rf_startup$rent_cpi_lag 
      median_price_current_is_previous_for_next <- last_observation_rf_startup$median_price
      
      rent_cpi_forecasts_rf_list_region <- list()
      
      for (i in 1:num_forecast_periods) {
        current_future_date_rf <- future_dates_for_forecast[i] 
        
        year_fut_rf = lubridate::year(current_future_date_rf); month_fut_rf = lubridate::month(current_future_date_rf)
        quarter_fut_rf = lubridate::quarter(current_future_date_rf); half_year_fut_rf = ifelse(month_fut_rf %in% 1:6, 1, 2)
        
        median_price_fut_rf = future_median_prices_for_rf_loop[i] 
        log_median_price_fut_rf = ifelse(!is.na(median_price_fut_rf) & median_price_fut_rf > 0, log(median_price_fut_rf), NA)
        
        price_growth_monthly_fut_rf <- ifelse(!is.na(median_price_current_is_previous_for_next) & median_price_current_is_previous_for_next != 0 & !is.na(median_price_fut_rf), (median_price_fut_rf - median_price_current_is_previous_for_next) / median_price_current_is_previous_for_next, NA)
        if (i == 1 && is.na(price_growth_monthly_fut_rf) && !is.na(last_observation_rf_startup$price_growth_monthly)) price_growth_monthly_fut_rf <- last_observation_rf_startup$price_growth_monthly
        
        rented_ratio_national_fut_rf <- last_national_ratios_fc$rented_ratio_national
        owner_ratio_national_fut_rf <- last_national_ratios_fc$owner_ratio_national
        free_ratio_national_fut_rf <- last_national_ratios_fc$free_ratio_national
        
        rent_cpi_lag_fut <- rent_cpi_current_is_previous_for_next 
        
        rent_growth_monthly_fut_rf <- ifelse(!is.na(rent_cpi_current_is_previous_for_next) & !is.na(rent_cpi_previous_is_two_ago_for_next) & rent_cpi_previous_is_two_ago_for_next != 0, (rent_cpi_current_is_previous_for_next - rent_cpi_previous_is_two_ago_for_next) / rent_cpi_previous_is_two_ago_for_next, NA)
        if (i == 1 && is.na(rent_growth_monthly_fut_rf) && !is.na(last_observation_rf_startup$rent_growth_monthly) ) rent_growth_monthly_fut_rf <- last_observation_rf_startup$rent_growth_monthly
        
        price_rent_cpi_ratio_fut_approx_rf <- ifelse(rent_cpi_current_is_previous_for_next > 0 & !is.na(median_price_fut_rf), median_price_fut_rf / rent_cpi_current_is_previous_for_next, NA)
        
        future_row_for_rf_predict <- tibble(
          date = current_future_date_rf, region = current_forecast_region, median_price = median_price_fut_rf,
          rented_ratio_national = rented_ratio_national_fut_rf, owner_ratio_national = owner_ratio_national_fut_rf,
          free_ratio_national = free_ratio_national_fut_rf, price_rent_cpi_ratio = price_rent_cpi_ratio_fut_approx_rf, 
          log_median_price = log_median_price_fut_rf, price_growth_monthly = price_growth_monthly_fut_rf,
          rent_growth_monthly = rent_growth_monthly_fut_rf, year = year_fut_rf, month = month_fut_rf, 
          quarter = quarter_fut_rf, half_year = half_year_fut_rf
        )
        
        predicted_rent_cpi_value_rf <- predict(active_forecasting_model_rf, newdata = future_row_for_rf_predict)
        
        current_forecast_row_rf <- tibble(date = current_future_date_rf, region = current_forecast_region, rent_cpi_forecast = as.numeric(predicted_rent_cpi_value_rf))
        rent_cpi_forecasts_rf_list_region[[i]] <- current_forecast_row_rf
        
        rent_cpi_previous_is_two_ago_for_next <- rent_cpi_current_is_previous_for_next       
        rent_cpi_current_is_previous_for_next <- as.numeric(predicted_rent_cpi_value_rf) 
        median_price_current_is_previous_for_next <- median_price_fut_rf 
      } # End of inner forecasting loop for RF
      
      # Store the RF forecast for the current region
      all_regional_rf_cpi_forecasts[[current_forecast_region]] <- bind_rows(rent_cpi_forecasts_rf_list_region)
      print(paste("Recursive RF forecasting complete for", current_forecast_region))
      
      # Plotting RF forecast for the current region
      historical_rent_cpi_for_plot_rf <- ml_data_final %>% filter(region == current_forecast_region) %>% select(date, rent_cpi)
      current_rf_plot_data <- bind_rows(
        historical_rent_cpi_for_plot_rf %>% rename(value = rent_cpi) %>% mutate(type = "Actual"),
        all_regional_rf_cpi_forecasts[[current_forecast_region]] %>% rename(value = rent_cpi_forecast) %>% mutate(type = "Forecast (RF Recursive)")
      )
      if(nrow(current_rf_plot_data) > 0) {
        rf_plot <- ggplot(current_rf_plot_data, aes(x = date, y = value, color = type, group = type)) + geom_line(linewidth = 1) +
          scale_colour_manual(values = c("Actual" = "navyblue", "Forecast (RF Recursive)" = "darkgoldenrod1"), name = "Data Type:") +
          scale_y_continuous(labels = comma) +
          labs(title = paste("Rent CPI RF Forecast:", current_forecast_region), subtitle = "Recursive using ARIMA for Median Price input", y = "Rent CPI") +
          theme(legend.position = "top") 
        print(rf_plot)
      }
    } # End else for last_observation_rf_startup check
  } # End else for future_median_prices_for_rf_loop check
} # --- End of Loop Through Regions ---

print("--- All Regional Forecasting Attempts Completed ---")

# --- 6. Combine All Forecasts for Overall Plot (Optional) ---
# This combines all ARIMA forecasts and all RF forecasts into two dataframes
final_arima_forecasts_all_regions <- bind_rows(all_regional_arima_cpi_forecasts)
final_rf_forecasts_all_regions <- bind_rows(all_regional_rf_cpi_forecasts)

if(nrow(final_arima_forecasts_all_regions) > 0 && nrow(final_rf_forecasts_all_regions) > 0) {
  print("Generating combined forecast plot for all processed regions...")
  
  # Prepare historical data for all regions
  historical_data_all_regions <- full_data %>%
    filter(region %in% unique_regions_to_forecast) %>%
    select(date, region, rent_cpi) %>%
    rename(value = rent_cpi) %>%
    mutate(method = "Actual", type = "Actual")
  
  # Combine forecasts
  comparison_plot_data_all <- bind_rows(
    historical_data_all_regions,
    final_arima_forecasts_all_regions %>% 
      rename(value = rent_cpi_forecast_arima) %>% 
      mutate(method = "ARIMA Direct", type = "Forecast (ARIMA)"),
    final_rf_forecasts_all_regions %>% 
      rename(value = rent_cpi_forecast) %>% 
      mutate(method = "RF Recursive", type = "Forecast (RF Recursive)")
  ) %>% filter(!is.na(value)) # Ensure no NA values interfere with plotting
  
  comparison_plot_data_all$type <- factor(comparison_plot_data_all$type, levels = c("Actual", "Forecast (ARIMA)", "Forecast (RF Recursive)"))
  
  all_regions_forecast_plot <- ggplot(comparison_plot_data_all, aes(x = date, y = value, colour = type, linetype = type)) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~region, scales = "free_y", ncol = 2) + # Facet by region
    scale_colour_manual(values = c("Actual" = "black", "Forecast (ARIMA)" = "deepskyblue3", "Forecast (RF Recursive)" = "orangered2"), name = "Data/Method:") +
    scale_linetype_manual(values = c("Actual" = "solid", "Forecast (ARIMA)" = "dashed", "Forecast (RF Recursive)" = "dotdash"), name = "Data/Method:") +
    scale_y_continuous(labels = comma) +
    labs(title = "Rent CPI Forecast Comparison Across Regions",
         subtitle = "ARIMA Direct vs. Recursive Random Forest (with ARIMA for Median Price input)",
         x = "Date", y = "Rent CPI") +
    theme(legend.position = "top", legend.key.width = unit(1.5, "cm"),
          strip.text = element_text(size=rel(0.8)))
  print(all_regions_forecast_plot)
  
} else {
  print("Could not generate combined forecast plot; not enough forecast data from both methods for all regions.")
}


# --------------------------------------------------------------------------
# End of Script
# --------------------------------------------------------------------------

