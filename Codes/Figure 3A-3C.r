# Load required libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(scales)

# Define theme settings
main_theme <- theme_bw() + 
  theme(
    legend.position = "none",
    plot.title = element_text(size = 28),
    axis.title.x = element_text(size = 0, color = "black", hjust = 0.5),
    axis.title.y = element_text(size = 0, color = "black", hjust = 0.5),
    axis.ticks = element_line(color = "black"), 
    axis.text.y = element_text(color = "black", size = 20),
    axis.text.x = element_text(color = "black", size = 0)
  )

# Define color palettes
line_colors <- c('grey70', '#669bbc', '#cdb4db')
bar_colors <- c('#F6A071', '#8ecae6', "#cdb4db", 'grey')

# Data loading function
load_model_data <- function(base_path) {
  # Load data for each model
  pls <- read.csv(paste0(base_path, '/pls-da_1/pre.csv'), sep = ",", header = TRUE, row.names = NULL)
  pls$model <- 'pls-da'
  
  ridge <- read.csv(paste0(base_path, '/ridge_1/pre.csv'), sep = ",", header = TRUE, row.names = NULL)
  ridge$model <- 'ridge'
  
  rf <- read.csv(paste0(base_path, '/rf_1/pre.csv'), sep = ",", header = TRUE, row.names = NULL)
  rf$model <- 'rf'
  
  lstm <- read.csv(paste0(base_path, '/lstm_1/pre.csv'), sep = ",", header = TRUE, row.names = NULL)
  lstm$model <- 'lstm'
  
  # Combine all data
  pre <- rbind(pls, ridge, rf, lstm)
  
  return(pre)
}

# Data preprocessing function
preprocess_data <- function(data, model_filter = 'pls-da', feature_filter = c('spc', 'spc_de', 'spc_rmR4')) {
  # Filter data
  pre_sub <- data %>% 
    filter(model == model_filter) %>% 
    filter(feature %in% feature_filter)
  
  # Set factor levels for time_group
  pre_sub$time_group <- factor(pre_sub$time_group, 
                              levels = c('t_>10', 't_10-8', 't_7-6', 't_5-4', 't_3', 't_2', 't_1', 't_0'))
  
  # Set factor levels for feature
  pre_sub$feature <- factor(pre_sub$feature, 
                           levels = c('spc', 'spc_rmR4', 'spc_de'))
  
  return(pre_sub)
}

# Create line plot function
create_line_plot <- function(data, y_var, y_min, y_max, title = "") {
  ggplot(data, aes(x = time_group, y = get(y_var), group = feature, color = feature)) +
    geom_line(size = 2.5) +
    geom_point(size = 4) +
    geom_ribbon(aes(ymin = get(paste0(y_var, "_ci_low")), 
                    ymax = get(paste0(y_var, "_ci_up")), 
                    fill = feature), 
                alpha = 0.2, color = NA) +
    scale_y_continuous(limits = c(0, 1),
                      labels = scales::number_format(accuracy = 0.1)) +
    scale_color_manual(values = line_colors) +
    scale_fill_manual(values = line_colors) +
    ylim(y_min, y_max) +
    main_theme
}

# Create bar plot function
create_bar_plot <- function(data, y_var, feature_colors = bar_colors, show_legend = FALSE) {
  # Set theme based on legend visibility
  plot_theme <- main_theme
  if (show_legend) {
    plot_theme <- main_theme + theme(legend.position = "right")
  }
  
  ggplot(data, aes(x = time_group, y = get(y_var), fill = feature)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), 
             alpha = 0.8, width = 0.6, color = '#adb5bd') +
    geom_errorbar(aes(ymin = get(paste0(y_var, "_ci_low")), 
                      ymax = get(paste0(y_var, "_ci_up"))),
                  position = position_dodge(width = 0.8),
                  width = 0.25,
                  color = '#adb5bd') +
    scale_y_continuous(limits = c(0, 1),
                      labels = scales::number_format(accuracy = 0.1)) +
    scale_fill_manual(values = feature_colors) +
    plot_theme + 
    coord_flip()
}

# Create model comparison bar plot
create_model_comparison_plot <- function(data, y_var) {
  ggplot(data, aes(x = time_group, y = get(y_var), fill = model)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), 
             alpha = 0.8, width = 0.6, color = '#adb5bd') +
    geom_errorbar(aes(ymin = get(paste0(y_var, "_ci_low")), 
                      ymax = get(paste0(y_var, "_ci_up"))),
                  position = position_dodge(width = 0.8),
                  width = 0.25,
                  color = '#adb5bd') +
    scale_y_continuous(limits = c(0, 1),
                      labels = scales::number_format(accuracy = 0.1)) +
    scale_fill_manual(values = bar_colors) +
    main_theme + 
    coord_flip()
}

# Main execution function
main <- function() {
  # Set plot dimensions
  options(repr.plot.width = 9, repr.plot.height = 2)
  
  # Load data
  base_path <- '.'
  pre <- load_model_data(base_path)
  
  # Create line plots for different metrics
  # 1. AUC plot
  pre_sub <- preprocess_data(pre)
  auc_plot <- create_line_plot(pre_sub, "outer_auc_mean", 0.5, 0.83)
  
  # 2. Accuracy plot
  acc_plot <- create_line_plot(pre_sub, "outer_acc_mean", 0.5, 0.83)
  
  # 3. Sensitivity plot
  sen_plot <- create_line_plot(pre_sub, "outer_sen_mean", 0.3, 0.80)
  
  # 4. Specificity plot
  spc_plot <- create_line_plot(pre_sub, "outer_spc_mean", 0.5, 0.83)
  
  # Create bar plots for feature comparison
  pre_sub_feature <- preprocess_data(pre, 
                                    model_filter = 'pls-da', 
                                    feature_filter = c('spc', 'spc+my', 'spc+scc', 'spc+parity'))
  pre_sub_feature$time_group <- factor(pre_sub_feature$time_group, 
                                      levels = c('t_0', 't_1', 't_2', 't_3', 't_5-4', 't_7-6', 't_10-8', 't_>10'))
  pre_sub_feature$feature <- factor(pre_sub_feature$feature, 
                                   levels = c('spc+parity', 'spc+scc', 'spc+my', 'spc'))
  
  options(repr.plot.width = 4, repr.plot.height = 5)
  feature_bar_plot <- create_bar_plot(pre_sub_feature, "outer_auc_mean", show_legend = TRUE)
  
  # Create model comparison plot
  pre_sub_model <- pre %>% filter(feature == 'spc')
  pre_sub_model$time_group <- factor(pre_sub_model$time_group, 
                                    levels = c('t_0', 't_1', 't_2', 't_3', 't_5-4', 't_7-6', 't_10-8', 't_>10'))
  pre_sub_model$model <- factor(pre_sub_model$model, 
                               levels = c('lstm', 'rf', 'ridge', 'pls-da'))
  
  model_bar_plot <- create_model_comparison_plot(pre_sub_model, "outer_auc_mean")
  
  # Return all plots
  return(list(
    auc_plot = auc_plot,
    acc_plot = acc_plot,
    sen_plot = sen_plot,
    spc_plot = spc_plot,
    feature_bar_plot = feature_bar_plot,
    model_bar_plot = model_bar_plot
  ))
}

# Run the script
results <- main()

# Display results
print(results$auc_plot)
print(results$acc_plot)
print(results$sen_plot)
print(results$spc_plot)
print(results$feature_bar_plot)
print(results$model_bar_plot)
