# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(patchwork)

# Define theme settings
main_theme <- theme_linedraw() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 0),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 30),
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

# Data loading function
load_data <- function(file_path) {
  data <- read.csv(file_path, sep = ",", header = TRUE, row.names = NULL)
  data <- subset(data, select = c('disease', 'disease_in', 'milkweightlbs', 'parity', 'cells', 'conductivity'))
  data <- na.omit(data)
  return(data)
}

# Data processing function
process_data <- function(data) {
  # Convert disease to factor
  data$disease <- ifelse(data$disease == 1, "Disease", "Health")
  data$disease <- factor(data$disease, levels = c("Health", "Disease"))
  
  # Process disease_in variable
  data <- data %>%
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
  
  # Set factor levels for disease_in
  data$disease_in <- factor(data$disease_in, 
                            levels = c("healthy", ">10", "10-8", "7-6", 
                                       "5-4", "3", "2", "1", "0"))
  
  return(data)
}

# Create violin-box plot function
create_violin_box_plot <- function(data, y_var, title) {
  ggplot(data, aes(x = disease, y = !!sym(y_var), fill = disease)) + 
    geom_violin() +
    geom_boxplot(width = 0.5, fill = "white") +
    stat_summary(fun.y = mean, geom = "point", shape = 23, size = 4) + 
    scale_fill_manual(values = c('grey', '#F6A071')) +
    ggtitle(title) +
    main_theme
}

# Create box plot function
create_box_plot <- function(data, y_var, log_transform = FALSE) {
  y_expr <- if(log_transform) paste0("log2(", y_var, ")") else y_var
  
  ggplot(data, aes(x = disease_in, y = !!sym(y_var), fill = disease_in)) + 
    geom_boxplot(width = 0.5, alpha = 0.8) +
    stat_summary(fun.y = median, 
                 geom = 'line', 
                 color = 'black', 
                 aes(group = disease), 
                 position = position_dodge(width = 0.9), 
                 size = 1) +
    scale_color_manual(values = color_palette) +
    scale_fill_manual(values = color_palette) +
    theme_linedraw() +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 28),
      axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
      axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
      axis.ticks = element_line(color = "black"),
      axis.text.y = element_text(color = "black", size = 10),
      axis.text.x = element_text(color = "black", size = 0)
    )
}

# Statistical test function
perform_statistical_tests <- function(data, y_var, group_var) {
  # Perform pairwise Wilcoxon tests with BH correction
  corrected <- as.data.frame(
    pairwise.wilcox.test(
      data[[y_var]], 
      data[[group_var]], 
      p.adjust.method = 'BH'
    )$p.value
  )
  
  # Perform pairwise Wilcoxon tests without correction
  uncorrected <- as.data.frame(
    pairwise.wilcox.test(
      data[[y_var]], 
      data[[group_var]], 
      p.adjust.method = "none"
    )$p.value
  )
  
  # Format p-values to 4 decimal places
  df_uncorrected <- corrected
  df_uncorrected[] <- as.data.frame(
    lapply(uncorrected, function(x) sprintf("%.4f", x))
  )
  
  df_corrected <- corrected
  df_corrected[] <- as.data.frame(
    lapply(corrected, function(x) sprintf("%.4f", x))
  )
  
  # Combine uncorrected and corrected p-values
  p_values <- as.data.frame(
    mapply(function(x, y) paste(x, y, sep="/"), 
           df_uncorrected, df_corrected)
  )
  
  return(list(
    corrected = corrected,
    uncorrected = uncorrected,
    combined = p_values
  ))
}

# Main execution function
main <- function() {
  # Load and process data
  file_path <- './JM006_spc_up.csv'
  data <- load_data(file_path)
  processed_data <- process_data(data)
  
  # Create plots for disease comparison
  p1 <- create_violin_box_plot(processed_data, "milkweightlbs", "Milk yield")
  p2 <- create_violin_box_plot(processed_data, "log2(cells)", "Somatic cell count")
  p3 <- create_violin_box_plot(processed_data, "parity", "Parity")
  
  # Adjust plot margins
  p1_adjusted <- p1 + theme(plot.margin = margin(b = 30, unit = "pt"))
  p2_adjusted <- p2 + theme(plot.margin = margin(t = 30, unit = "pt"))
  
  # Combine plots
  combined_plot <- p1_adjusted + p2_adjusted + plot_layout(nrow = 2)
  
  # Create box plots for disease_in comparison
  milk_plot <- create_box_plot(processed_data, "milkweightlbs")
  scc_plot <- create_box_plot(processed_data, "cells", log_transform = TRUE)
  
  # Perform statistical tests
  milk_tests <- perform_statistical_tests(processed_data, "milkweightlbs", "disease_in")
  scc_tests <- perform_statistical_tests(processed_data, "cells", "disease_in")
  
  # Return results
  return(list(
    combined_plot = combined_plot,
    milk_plot = milk_plot,
    scc_plot = scc_plot,
    milk_tests = milk_tests,
    scc_tests = scc_tests
  ))
}

# Run the script
results <- main()

# Display results
print(results$combined_plot)
print(results$milk_plot)
print(results$scc_plot)
print("Milk yield statistical tests:")
print(results$milk_tests$combined)
print("SCC statistical tests:")
print(results$scc_tests$combined)
