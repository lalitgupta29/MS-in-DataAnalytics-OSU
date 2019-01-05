  # load libraries
  library(ggplot2)
  
  # set seed so we get consistent results on every run
  set.seed(2018)
  
  # ==   Simulation Study    =====
  
  # load the data
  yrbss_2003 <- readRDS("yrbss_2003.rds")
  yrbss_2013 <- readRDS("yrbss_2013.rds")
  
  n_sim <- 10000 # number of simulations
  
  
  
  # ==  Simulation Study: Mean     =====
  
  
  # extract and save the variable of interest so they 
  # can be accessed easily
  bmi_2003 <- yrbss_2003$bmi
  bmi_2013 <- yrbss_2013$bmi
  
  # Let's look at the Population (bmi_2013) for BMI Index
  # plotting the population
  qplot(bmi_2013, binwidth = 1) +
    ggtitle("BMI Index for High-school student in 2013") +
    xlab("BMI Index") +
    ylab("No. of Students") +
    geom_vline(xintercept = mean(bmi_2013), color = "red") +
    geom_vline(xintercept = median(bmi_2013), color = "green")
  
  # Population (bmi_2013) parameters
  cbind(Pop_Mean = round(mean(bmi_2013),3), 
        Pop_SD = round(sd(bmi_2013),3), 
        Pop_Median = round(median(bmi_2013), 3),
        Pop_quantile = round(quantile(bmi_2013, probs = 0.25), 3)
  )
  
  # Writing a function to get means of random samples of  
  # size n from population x
  get_mean <- function(n, n_sim, x = bmi_2013){
    replicate(n_sim, mean(sample(x, n, replace=FALSE)))
  }
  
  # get means for n_sim repeated random samples of size 10,  
  # 100 and 1000
  ns <- c(10, 100, 1000)
  means <- lapply(ns, get_mean, n_sim = n_sim)
  
  # Let's see how the sampling distribution looks like 
  # for sample size 10 and 1000
  qplot(means[[1]], binwidth = 0.2) + 
    ggtitle("Sampling Distribution (sample size 10)") + 
    xlab("Mean BMI") +
    ylab("No. of Samples")
  qplot(means[[3]], binwidth = 0.05) + 
    ggtitle("Sampling Distribution (sample size 1000)") + 
    xlab("Mean BMI") +
    ylab("No. of Samples")
  
  # Let's get the mean and SD for our sampling Distributions
  mean_sam_dist <- sapply(means, mean)  
  sd_sam_dist <- sapply(means, sd)  
  cbind(SampleSize = ns, 
        Mean = round(mean_sam_dist, 3), 
        SD = round(sd_sam_dist, 3))
  
  
  
  # == Simulation Study: 25 percentile  =====
  
  
  # Writing function to get 25% quantile of random samples of  
  # size n from population x
  get_quantile <- function(n, n_sim, x = bmi_2013){
    replicate(n_sim, quantile((sample(x, n, replace=FALSE)), 
                              probs = 0.25))
  }
  
  # get 25% quantiles for n_sim repeated random samples of size  
  # 10, 100 and 1000
  ns <- c(10, 100, 1000)
  quantiles <- lapply(ns, get_quantile, n_sim = n_sim)
  
  # Let's see how the sampling distribution looks like 
  # for sample size 10 and 1000
  qplot(quantiles[[1]], binwidth = 0.2) + 
    ggtitle("Sampling Distribution (sample size 10)") + 
    xlab("25% quantile") +
    ylab("No. of Samples")
  qplot(quantiles[[3]], binwidth = 0.05) + 
    ggtitle("Sampling Distribution (sample size 1000)") + 
    xlab("25% quantile") +
    ylab("No. of Samples")
  
  # Let's get the mean, median and SD for our sampling Distributions
  mean_quantile <- sapply(quantiles, mean)
  median_quantile <- sapply(quantiles, median)
  sd_quantile <- sapply(quantiles, sd)
  cbind(SampleSize = ns, 
        Mean = round(mean_quantile, 3),
        Median = round(median_quantile, 3),
        SD = round(sd_quantile, 3)
  )
  
  # Population (bmi_2013) parameters for BMI Index
  cbind(Pop_Quantile = quantile(bmi_2013, probs = 0.25))  
  
  
  
  # == Simulation Study: Minimum  =====
  
  
  # function to get minimum of random samples of size n 
  # from population x
  get_min <- function(n, n_sim, x = bmi_2013){
    replicate(n_sim, min((sample(x, n, replace=FALSE))))
  }
  
  # get minimum for n_sim repeated random samples of size 10, 
  # 100 and 1000
  ns <- c(10, 100, 1000)
  mins <- lapply(ns, get_min, n_sim = n_sim)
  
  # Let's see how the sampling distribution looks like 
  # for sample size 10 and 1000
  qplot(mins[[1]], binwidth = 0.4) + 
    ggtitle("Sampling Distribution (sample size 10)") + 
    xlab("Sample Minimum") +
    ylab("No. of Samples")
  qplot(mins[[3]], binwidth = 0.2) + 
    ggtitle("Sampling Distribution (sample size 1000)") + 
    xlab("Sample Minimum") +
    ylab("No. of Samples")
  
  # Let's get the mean, median and sd for our sampling Distributions
  mean_mins <- sapply(mins, mean)
  median_mins <- sapply(mins, median)
  sd_mins <- sapply(mins, sd)
  cbind(SampleSize = ns, Mean = round(mean_mins, 3),
        Median = round(median_mins, 3),
        SD = round(sd_mins, 3))
  
  # Population (bmi_2013) parameters for BMI Index
  cbind(Pop_Minimun = min(bmi_2013))
  
  
  
  # == Simulation Study: Diffrence in Median  =====
  
  
  # function to get minimum of difference of sample median BMI 
  # between 2013 and 2003 by using sample size n1 and n2 respectively
  get_median_diff <- function(n1, n2, n_sim, x1 = bmi_2013, 
                              x2 = bmi_2003){
    median_2013 <- replicate(n_sim, median((sample(x1, n1, 
                                                   replace=FALSE))))
    median_2003 <- replicate(n_sim, median((sample(x2, n2, 
                                                   replace=FALSE))))
    median_2013-median_2003
  }
  
  # get Difference in sample medians between 2013 and 2003 for 
  # n_sim repeated random samples of size 5,5, 10,10 and 100,100 
  med_diffs <- list()
  med_diffs[[1]] <- get_median_diff(5, 5, n_sim)
  med_diffs[[2]] <- get_median_diff(10,10, n_sim)
  med_diffs[[3]] <- get_median_diff(100, 100, n_sim)
  
  # Let's see how the sampling distribution looks like 
  # for sample size 5,5 and 100,100
  qplot(med_diffs[[1]], binwidth = 1) + 
    ggtitle("Sampling Distribution (sample size 5, 5)") + 
    xlab("Difference in Sample Median") +
    ylab("No. of Samples")
  qplot(med_diffs[[3]], binwidth = 0.2) + 
    ggtitle("Sampling Distribution (sample size 100, 100)") + 
    xlab("Difference in Sample Median") +
    ylab("No. of Samples")
  
  # Let's get the mean and SD for our sampling Distributions
  mean_med_diff <- sapply(med_diffs, mean)
  sd_med_diff <- sapply(med_diffs, sd)
  cbind(SampleSize = c('n1=5,n2=5', 'n1=10,n2=10', 
                       'n1=100,n2=100'), 
        Mean_Diff = round(mean_med_diff, 3), 
        SD_Diff = round(sd_med_diff, 3))
  
  # Population (bmi_2013 and bmi_2003 ) parameters for BMI Index
  median(bmi_2013) - median(bmi_2003)  
  
  
  
  # == Data Analysis: BMI =====
  
  
  # let's look the variables of interest
  qplot(yrbss_2013$bmi, binwidth = 1) +
    ggtitle("BMI Index for High-school student in 2013") +
    xlab("BMI Index") +
    ylab("No. of Students") +
    geom_vline(xintercept = mean(bmi_2013), color = "red") +
    geom_vline(xintercept = median(bmi_2013), color = "green")
  
  qplot(yrbss_2003$bmi, binwidth = 1) +
    ggtitle("BMI Index for High-school student in 2003") +
    xlab("BMI Index") +
    ylab("No. of Students") +
    geom_vline(xintercept = mean(bmi_2013), color = "red") +
    geom_vline(xintercept = median(bmi_2013), color = "green")
  
  # perform two sample t-test to see if the BMI changes are  
  # significant enough to draw statistical significance.
  t.test(bmi_2013, bmi_2003, mu = 0, alternative = "two.sided", 
         paired = FALSE, var.equal = FALSE)
  
  
  
  # == Data Analysis: Smoking Habits   =====
  
  
  # let's look at the variable of interest
  smoke <- table(yrbss_2013$sex, yrbss_2013$q33)
  bp <- barplot(smoke, main = "Smokers (Male vs Female)",
          legend = c("Female","Male"), beside = TRUE, 
          col=heat.colors(2), ylim = c(0,6000),
          names.arg = c("0", "1-2", "3-5", 
                        "6-9","10-19", "20-29",
                        "All 30"),
          xlab = "No. of Days Smoked")
  text(bp, smoke, smoke, cex = 0.8, pos = 3, offset = 0.2)
  
  # let's perform a one sided proportions test with significance
  # level 0.05 and alternative hypothesis that population proportion
  # of male high-schoolers who smoke is higher than population 
  # proportion of female high-schoolers
  
  # converting qualitative variables to quantitative variables:
  # anyone smoking less than 20 days is considered non-smoker;
  # considered smoker otherwise
  yrbss_2013$smokers <- as.character(yrbss_2013$q33)
  yrbss_2013$smokers[yrbss_2013$smokers %in% c("0 days", 
                                       "1 or 2 days", 
                                       "3 to 5 days",
                                       "6 to 9 days",
                                       "10 to 19 days")] <- 0
  yrbss_2013$smokers[yrbss_2013$smokers %in% c("20 to 29 days", 
                                       "All 30 days")] <- 1
  
  # performing one sided proportion test
  sex_smoke_table <- table(yrbss_2013$smokers, yrbss_2013$sex)
  prop.test(sex_smoke_table, conf.level = 0.95, alternative = "g", 
            correct = FALSE)
  # performing two sided proportions test to get confidence intervals
  prop.test(sex_smoke_table, conf.level = 0.95, alternative = "t", 
            correct = FALSE)
  
  # Let's see if the result is different when we classify smokers
  # more rigidly, i.e. anyone smoking for more than 0 days is
  # classified as smoker
  yrbss_2013$smokers <- as.character(yrbss_2013$q33)
  yrbss_2013$smokers[yrbss_2013$smokers %in% c("0 days")] <- 0
  yrbss_2013$smokers[yrbss_2013$smokers %in% c("1 or 2 days",
                                       "3 to 5 days","6 to 9 days",
                                       "10 to 19 days", 
                                       "20 to 29 days", 
                                       "All 30 days")] <- 1
  
  # performing one sided proportion test
  sex_smoke_table <- table(yrbss_2013$smokers, yrbss_2013$sex)
  prop.test(sex_smoke_table, conf.level = 0.95, alternative = "g", 
            correct = FALSE)
  # performing two sided proportions test to get confidence intervals
  prop.test(sex_smoke_table, conf.level = 0.95, alternative = "t", 
            correct = FALSE)
  
  
  
  # == Data Analysis: TV Time  =====
  
  
  # Let's explore the variable of interest
  tv_time <- table(yrbss_2013$q81)
  
  # let's summarize the variable with a bar graph
  bp <- barplot(tv_time, main = "TV Time per day (School-days)", 
                col = heat.colors(7), ylab = "No. of Students",
                names.arg = c("0", "< 1", "1", "2", 
                              "3", "4", ">= 5"), width = 1.2,
                args.legend = list(title = "TV Time", x = "topright", 
                                   cex = .7),
                xlab = "No. of Hours per day",
                beside = TRUE, ylim = c(0, 2700))
  tv_time_percent <- as.character(round((tv_time*100/sum(tv_time)),2))
  tv_time_percent <- paste(tv_time_percent, "%", sep = "")
  text(bp, 0, tv_time, cex=0.9, pos=3, offset = 0.2)
  text(bp, tv_time, tv_time_percent, cex=0.9, pos=1, offset = 0.3) 
  
  # Let's see how TV time differs between male and female students
  tv_time_sex <- table(yrbss_2013$q81, yrbss_2013$sex)
  
  bp <- barplot(tv_time_sex, main = "TV Time per day (School-days)", 
                col = heat.colors(7), xlab = "Gender", 
                names = c("Male", "Female"), beside = TRUE, 
                ylab = "No. of Students", ylim = c(0,1500),
                legend = c("0 Hr", "< 1 Hr", "1 Hr", "2 Hr", "3 Hr", 
                           "4 Hr", ">= 5 Hr"), 
                args.legend = list(title = "TV Time", x = "topright", 
                                   cex = 0.48))
  text(bp, y = tv_time_sex, tv_time_sex, cex = 0.9, pos=3, 
       offset = 0.2)
  lines(x = bp, y = tv_time_sex)
  
  # Let's see if watching more TV has any correlation with BMI 
  # index
  tv_bmi <- aggregate(yrbss_2013$bmi, list(yrbss_2013$q81), mean)
  colnames(tv_bmi) <- c("TV-Time", "bmi")
  
  bp <- barplot(tv_bmi$bmi, ylim = c(0,28), 
                xlab = "TV Time", ylab = "BMI", 
                col = heat.colors(7), width = 1.2,
                names.arg = c("0 Hr", "< 1 Hr", "1 Hr", 
                              "2 Hr", "3 Hr", "4 Hr", 
                              ">= 5 Hr"),
                main = "TV Time vs BMI")
  text(bp, y = tv_bmi$bmi, round(tv_bmi$bmi,2), cex = 0.7, 
       pos=3, offset = 0.2)
  
  # Time spent watching TV and BMI index do seem to be positively 
  # correlated. Let's perform a two sample t-test to find out if 
  # watching tv for more than 2 hours per day results in increased 
  # BMI. 
  yrbss_2013$tv <- as.character(yrbss_2013$q81)
  yrbss_2013$tv[yrbss_2013$tv %in% c("No TV on average school day",
                                     "Less than 1 hour per day",
                                     "1 hour per day",
                                     "2 hours per day")] <- 0
  yrbss_2013$tv[yrbss_2013$tv %in% c("3 hours per day",
                                     "4 hours per day",
                                     "5 or more hours per day")] <- 1
  
  tv_2hr_less <- yrbss_2013$bmi[yrbss_2013$tv == 0] 
  tv_2hr_more <- yrbss_2013$bmi[yrbss_2013$tv == 1]
  t.test(tv_2hr_more, tv_2hr_less, mu = 0, alternative = "t", 
         paired = FALSE, var.equal = FALSE)
  
  
