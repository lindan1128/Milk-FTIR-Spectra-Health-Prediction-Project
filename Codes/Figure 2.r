# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(grafify)
library(patchwork)

# Define theme settings
main_theme <- theme_minimal() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 20),
    axis.text.x = element_text(color = "black", size = 0)
  )

# Define color palette
color_palette <- c(
  "healthy" = "grey",
  ">10" = "#F6A071",
  "10-8" = "#F6A071",
  "7-6" = "#F6A071",
  "5-4" = "#F6A071",
  "3" = "#F6A071",
  "2" = "#F6A071",
  "1" = "#F6A071",
  "0" = "#F6A071"
)

# Data loading and preprocessing function
load_spectral_data <- function(file_path) {
  # Load spectral data
  spc <- read.csv(file_path, sep = ",", header = TRUE, row.names = 1)
  names(spc) <- gsub("^X", "", names(spc))
  
  # Filter spectral data
  spc_s <- spc
  spc_filtered <- spc_s[, which(as.numeric(colnames(spc_s)) > 1000 & as.numeric(colnames(spc_s)) < 3000)]
  spc_filtered <- spc_filtered[, which(as.numeric(colnames(spc_filtered)) < 1580 | as.numeric(colnames(spc_filtered)) > 1700)]
  spc_filtered$vial <- spc_s$vial
  spc_filtered <- cbind(spc_filtered, spc_s[, 937:953])
  
  return(spc_filtered)
}

# Calculate mean and SD for disease and health groups
calculate_group_statistics <- function(data) {
  # Disease group
  data_d <- data %>% filter(disease == 1)
  d_mean <- sapply(data_d, mean)
  d_sd <- sapply(data_d, sd)
  
  # Health group
  data_h <- data %>% filter(disease == 0)
  data_h <- data_h[, c(1:487)]
  h_mean <- sapply(data_h, mean)
  h_sd <- sapply(data_h, sd)
  
  # Combine statistics
  se <- data.frame(
    spc = c(as.numeric(colnames(data_d)), as.numeric(colnames(data_h))),
    mean = c(d_mean, h_mean),
    sd = c(d_sd, h_sd),
    group = c(rep('disease', length(d_sd)), rep('health', length(h_sd)))
  )
  
  return(se)
}

# Create spectral plot function
create_spectral_plot <- function(se, x_min, x_max, y_min, y_max, title = "") {
  pd <- position_dodge(0.1)
  
  ggplot(se, aes(x = spc, y = mean, color = group)) + 
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), size = 2, width = 4, alpha = 0.3) +
    geom_line(position = pd, size = 3) +
    scale_color_manual(values = c('#e76f51', 'grey')) +
    labs(x = 'Wavenumber', y = 'Absorbance', title = title) +
    xlim(x_min, x_max) +
    ylim(y_min, y_max) +
    main_theme
}

# Create violin plot function
create_violin_plot <- function(data, wavenumber, y_min = NULL, y_max = NULL) {
  # Extract data for specific wavenumber
  wav <- subset(data, select = c(wavenumber, 'disease_in', 'disease'))
  colnames(wav) <- c('value', 'disease_in', 'disease')
  
  # Process disease variable
  wav$disease <- ifelse(wav$disease == 1, "Disease", "Health")
  wav$disease <- factor(wav$disease, levels = c("Health", "Disease"))
  
  # Process disease_in variable
  wav <- wav %>%
    mutate(disease_in = case_when(
      disease_in == Inf ~ "healthy",
      disease_in > 10 & disease_in <= 30 ~ ">10",
      disease_in <= 10 & disease_in >= 8 ~ "10-8",
      disease_in <= 7 & disease_in >= 6 ~ "7-6",
      disease_in <= 5 & disease_in >= 4 ~ "5-4",
      disease_in == 3 ~ "3",
      disease_in == 2 ~ "2",
      disease_in == 1 ~ "1",
      disease_in == 0 ~ "0",
      TRUE ~ as.character(disease_in)
    ))
  
  # Set factor levels
  wav$disease_in <- factor(wav$disease_in, 
                            levels = c("healthy", ">10", "10-8", "7-6", 
                                       "5-4", "3", "2", "1", "0"))
  
  # Create plot
  p <- ggplot(wav, aes(x = disease_in, y = value, fill = disease_in)) + 
    geom_violin(alpha = 0.7) +
    stat_summary(fun.y = median, 
                 geom = 'line', 
                 color = '#F6A071', 
                 aes(group = disease), 
                 position = position_dodge(width = 0.9), 
                 size = 2) +
    scale_color_manual(values = color_palette) +
    scale_fill_manual(values = color_palette) +
    theme_linedraw() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 29),
      axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
      axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
      axis.text.y = element_text(color = "black", size = 25),
      axis.text.x = element_text(color = "black", size = 0)
    )
  
  # Add y-axis limits if provided
  if (!is.null(y_min) && !is.null(y_max)) {
    p <- p + ylim(y_min, y_max)
  }
  
  return(p)
}

# Perform statistical tests function
perform_statistical_tests <- function(data, wavenumber) {
  # Extract data for specific wavenumber
  wav <- subset(data, select = c(wavenumber, 'disease_in', 'disease'))
  colnames(wav) <- c('value', 'disease_in', 'disease')
  
  # Process disease variable
  wav$disease <- ifelse(wav$disease == 1, "Disease", "Health")
  wav$disease <- factor(wav$disease, levels = c("Health", "Disease"))
  
  # Process disease_in variable
  wav <- wav %>%
    mutate(disease_in = case_when(
      disease_in == Inf ~ "healthy",
      disease_in > 10 & disease_in <= 30 ~ ">10",
      disease_in <= 10 & disease_in >= 8 ~ "10-8",
      disease_in <= 7 & disease_in >= 6 ~ "7-6",
      disease_in <= 5 & disease_in >= 4 ~ "5-4",
      disease_in == 3 ~ "3",
      disease_in == 2 ~ "2",
      disease_in == 1 ~ "1",
      disease_in == 0 ~ "0",
      TRUE ~ as.character(disease_in)
    ))
  
  # Set factor levels
  wav$disease_in <- factor(wav$disease_in, 
                            levels = c("healthy", ">10", "10-8", "7-6", 
                                       "5-4", "3", "2", "1", "0"))
  
  # Perform tests
  corrected <- as.data.frame(as.matrix(
    (pairwise.wilcox.test(wav$value, wav$disease_in, p.adjust.method = 'BH')$p.value)
  ))
  
  uncorrected <- as.data.frame(as.matrix(
    (pairwise.wilcox.test(wav$value, wav$disease_in, p.adjust.method = 'none')$p.value)
  ))
  
  # Format p-values
  df_uncorrected <- corrected
  df_uncorrected[] <- as.data.frame(
    lapply(uncorrected, function(x) sprintf("%.4f", x))
  )
  
  df_corrected <- corrected
  df_corrected[] <- as.data.frame(
    lapply(corrected, function(x) sprintf("%.4f", x))
  )
  
  # Combine p-values
  p_values <- as.data.frame(
    mapply(function(x, y) paste(x, y, sep = "/"), 
           df_uncorrected, df_corrected)
  )
  
  return(list(
    corrected = corrected,
    uncorrected = uncorrected,
    combined = p_values
  ))
}

# Calculate statistics for specific wavenumbers
calculate_wavenumber_statistics <- function(data, wavenumber) {
  data_d <- data %>% filter(disease == 1)
  data_h <- data %>% filter(disease == 0)
  
  stats <- list(
    mean_d = mean(data_d[[wavenumber]]),
    mean_h = mean(data_h[[wavenumber]]),
    sd_d = sd(data_d[[wavenumber]]),
    sd_h = sd(data_h[[wavenumber]])
  )
  
  return(stats)
}

# Main execution function
main <- function() {
  # Load and process data
  file_path <- './JM006_spc_up.csv'
  spc_filtered <- load_spectral_data(file_path)
  
  # Calculate group statistics
  se <- calculate_group_statistics(spc_filtered)
  
  # Create spectral plots
  p1 <- create_spectral_plot(se, 1585, 1000, -0.1, 0.75)
  p2 <- create_spectral_plot(se, 1800, 1700, -0.1, 0.75)
  p3 <- create_spectral_plot(se, 3000, 2800, -0.1, 0.75)
  p4 <- create_spectral_plot(se, 2800, 1800, -0.1, 0.75)
  
  # Combine spectral plots
  combined_spectral_plot <- p3 + p4 + p2 + p1 + plot_layout(nrow = 1, widths = c(2, 1, 1, 5))
  
  # Create violin plots for specific wavenumbers
  wavenumbers <- c('2923.87', '2854.44', '1747.38', '1157.2', '1542.94', '1041.48')
  violin_plots <- list()
  
  for (wavenumber in wavenumbers) {
    if (wavenumber == '1542.94') {
      violin_plots[[wavenumber]] <- create_violin_plot(spc_filtered, wavenumber, 0, 0.5)
    } else {
      violin_plots[[wavenumber]] <- create_violin_plot(spc_filtered, wavenumber)
    }
  }
  
  # Perform statistical tests for each wavenumber
  test_results <- list()
  for (wavenumber in wavenumbers) {
    test_results[[wavenumber]] <- perform_statistical_tests(spc_filtered, wavenumber)
  }
  
  # Calculate statistics for specific wavenumbers
  wavenumber_stats <- list()
  for (wavenumber in wavenumbers) {
    wavenumber_stats[[wavenumber]] <- calculate_wavenumber_statistics(spc_filtered, wavenumber)
  }
  
  # Return results
  return(list(
    combined_spectral_plot = combined_spectral_plot,
    violin_plots = violin_plots,
    test_results = test_results,
    wavenumber_stats = wavenumber_stats
  ))
}

# Run the script
results <- main()

# Display results
print(results$combined_spectral_plot)

# Display violin plots
for (wavenumber in names(results$violin_plots)) {
  print(results$violin_plots[[wavenumber]])
}

# Display test results
for (wavenumber in names(results$test_results)) {
  print(paste("Test results for wavenumber:", wavenumber))
  print(results$test_results[[wavenumber]]$combined)
}

# Display wavenumber statistics
for (wavenumber in names(results$wavenumber_stats)) {
  print(paste("Statistics for wavenumber:", wavenumber))
  print(results$wavenumber_stats[[wavenumber]])
}
