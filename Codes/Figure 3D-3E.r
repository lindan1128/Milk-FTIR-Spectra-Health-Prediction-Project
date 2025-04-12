# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(scales)
library(patchwork)

# Define theme settings
main_theme <- theme_bw() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 29),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 10),
    axis.text.x = element_text(color = "black", size = 0)
  )

correlation_theme <- theme_linedraw() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 22),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"),
    axis.text.y = element_text(color = "black", size = 20),
    axis.text.x = element_text(color = "black", size = 20, angle = 75, vjust = 0.5)
  )

# Define color palettes
wave_group_colors <- c(
  "3000-2800" = "#8ecae6",
  "1800-1700" = "#8ecae6",
  "1585-1000" = "#cdb4db",
  "other" = "grey90"
)

# Function to process spectral data
process_spectral_data <- function(file_path) {
  # Load and preprocess spectral data
  spc <- read.csv(file_path, sep = ",", header = TRUE, row.names = 1)
  names(spc) <- gsub("^X", "", names(spc))
  
  # Filter columns based on wavenumber ranges
  cols <- colnames(spc)
  keep_cols <- cols[!is.na(cols) & 
                     cols >= 1000 & cols <= 3000 & 
                     !(cols >= 1580 & cols <= 1700) & 
                     !(cols >= 1800 & cols <= 2800)]
  spc_filtered <- spc[, keep_cols]
  
  # Calculate mean and standard deviation
  d_mean <- sapply(spc_filtered, mean)
  d_sd <- sapply(spc_filtered, sd)
  
  # Create data frame with processed data
  se <- data.frame(spc = as.numeric(colnames(spc_filtered)), mean = d_mean, sd = d_sd)
  
  # Group wavenumbers and create sequential numbering
  se_processed <- se %>%
    mutate(wave_group = case_when(
      spc >= 2800 & spc <= 3000 ~ "3000-2800",
      spc >= 1700 & spc <= 1800 ~ "1800-1700",
      spc >= 1000 & spc <= 1585 ~ "1585-1000",
      TRUE ~ "other"
    )) %>%
    filter(!is.na(wave_group)) %>%
    arrange(desc(wave_group), desc(spc)) %>%
    mutate(wavenumber = row_number())
  
  return(se_processed)
}

# Function to process importance data
process_importance_data <- function(imp_data, model_name, time_groups = c('t_3', 't_2', 't_1', 't_0')) {
  imp_sub <- imp_data %>% 
    filter(model == model_name) %>%
    filter(time_group %in% time_groups) %>%
    group_by(wavenumber) %>% 
    summarise(mean = mean(vip, na.rm = TRUE), 
              sd = sd(vip, na.rm = TRUE)) %>%
    mutate(original_wavenumber = wavenumber,
           wave_group = case_when(
             wavenumber >= 2800 & wavenumber <= 3000 ~ "3000-2800",
             wavenumber >= 1700 & wavenumber <= 1800 ~ "1800-1700",
             wavenumber >= 1000 & wavenumber <= 1585 ~ "1585-1000",
             TRUE ~ "other"
           ),
           alpha_value = ifelse(mean > 1, 1, 0.3)) %>%
    arrange(desc(wave_group), desc(original_wavenumber)) %>%
    mutate(wavenumber = row_number())
  
  return(imp_sub)
}

# Function to create VIP plot with spectral overlay
create_vip_plot <- function(imp_data, spectral_data, model_name, height = 2) {
  # Process importance data
  imp_sub <- process_importance_data(imp_data, model_name)
  
  # Set plot dimensions
  options(repr.plot.width = 9, repr.plot.height = height)
  
  # Create plot
  p <- ggplot() +
    # VIP bars
    geom_bar(data = imp_sub, 
             aes(x = wavenumber, y = mean, fill = wave_group, alpha = alpha_value),
             stat = "identity", position = "dodge") +
    geom_errorbar(data = imp_sub,
                  aes(x = wavenumber, ymin = mean - sd, ymax = mean + sd),
                  width = 0.2,
                  position = position_dodge(0.9)) +
    # Spectral line
    geom_line(data = spectral_data,
              aes(x = wavenumber, y = mean * max(imp_sub$mean) / max(mean)),
              color = "#F6A071", size = 0.7) +
    # Scales and theme
    scale_fill_manual(values = wave_group_colors) +
    scale_alpha_identity() +
    main_theme +
    coord_cartesian(expand = FALSE)
  
  return(p)
}

# Function to process data for correlation analysis
process_correlation_data <- function(spc_data, imp_data, model_name, time_groups = c('t_3', 't_2', 't_1', 't_0')) {
  # Filter data for healthy and disease groups
  spc_h <- spc_data %>% filter(disease == '0')
  spc_d <- spc_data %>% filter(disease == '1') %>% filter(disease_in <= 3)
  
  # Filter columns based on wavenumber ranges
  cols <- colnames(spc_data)
  keep_cols <- cols[!is.na(cols) & 
                     cols >= 1000 & cols <= 3000 & 
                     !(cols >= 1580 & cols <= 1700) & 
                     !(cols >= 1800 & cols <= 2800)]
  
  # Filter and calculate means
  spc_h_filtered <- spc_h[, keep_cols]
  spc_d_filtered <- spc_d[, keep_cols]
  
  # Calculate importance mean
  imp_mean <- imp_data %>% 
    filter(model == model_name) %>% 
    filter(time_group %in% time_groups) %>%
    group_by(wavenumber) %>% 
    summarise(mean = mean(vip, na.rm = TRUE))
  
  # Create correlation dataframe
  imps <- data.frame(
    spc = abs(colMeans(spc_h_filtered, na.rm = TRUE) - colMeans(spc_d_filtered, na.rm = TRUE)),
    imp = imp_mean$mean
  )
  
  return(imps)
}

# Function to create correlation plot
create_correlation_plot <- function(correlation_data) {
  p <- ggplot(correlation_data, aes(x = spc, y = imp)) + 
    geom_point(color = "grey", size = 3) +
    geom_smooth(method = lm, color = '#a8dadc', size = 2) +
    xlab('Difference in absorbance on each wavenumber between two groups') +
    ylab('Importance on each wavenumber') +
    correlation_theme
  
  # Calculate and print correlation statistics
  model <- lm(spc ~ imp, data = correlation_data)
  print(summary(model))
  
  return(p)
}

# Main execution function
main <- function() {
  # Load spectral data
  spectral_data <- process_spectral_data('./JM006_spc_up.csv')
  
  # Create VIP plots for each model
  pls_plot <- create_vip_plot(imp, spectral_data, 'pls-da', height = 1)
  ridge_plot <- create_vip_plot(imp, spectral_data, 'ridge')
  rf_plot <- create_vip_plot(imp, spectral_data, 'rf')
  lstm_plot <- create_vip_plot(imp, spectral_data, 'lstm')
  
  # Load data for correlation analysis
  spc_full <- read.csv('./JM006_spc_up.csv', sep = ",", header = TRUE, row.names = 1)
  names(spc_full) <- gsub("^X", "", names(spc_full))
  
  # Create correlation plots for each model
  pls_corr_data <- process_correlation_data(spc_full, imp, 'pls-da')
  ridge_corr_data <- process_correlation_data(spc_full, imp, 'ridge')
  rf_corr_data <- process_correlation_data(spc_full, imp, 'rf')
  lstm_corr_data <- process_correlation_data(spc_full, imp, 'lstm')
  
  pls_corr_plot <- create_correlation_plot(pls_corr_data)
  ridge_corr_plot <- create_correlation_plot(ridge_corr_data)
  rf_corr_plot <- create_correlation_plot(rf_corr_data)
  lstm_corr_plot <- create_correlation_plot(lstm_corr_data)
  
  # Set dimensions for correlation plots
  options(repr.plot.width = 12, repr.plot.height = 3)
  
  # Combine correlation plots
  correlation_combined <- pls_corr_plot + ridge_corr_plot + rf_corr_plot + lstm_corr_plot + 
    plot_layout(nrow = 1)
  
  # Return all plots
  return(list(
    vip_plots = list(
      pls = pls_plot,
      ridge = ridge_plot,
      rf = rf_plot,
      lstm = lstm_plot
    ),
    correlation_plots = list(
      pls = pls_corr_plot,
      ridge = ridge_corr_plot,
      rf = rf_corr_plot,
      lstm = lstm_corr_plot,
      combined = correlation_combined
    )
  ))
}

# Run the script
results <- main()

# Display results
print(results$vip_plots$pls)
print(results$vip_plots$ridge)
print(results$vip_plots$rf)
print(results$vip_plots$lstm)
print(results$correlation_plots$combined)
