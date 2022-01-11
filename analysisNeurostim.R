library(dplyr)
library(ggpubr)
library(tidyr)
library(rstatix)


data_folders=c("PSE_testFreqs.1995JansenRit_NEMOS-m08d12y2021-t18h.37m.01s",
               "PSE_testFreqs_wX2.1995JansenRit_NEMOS-m07d19y2021-t10h.38m.33s")

Results = data.frame()

for (folder in data_folders){
  
  weight = substring(folder, 16,17)
  if (weight == "99"){
    weight = "X1"
  }

  setwd(paste("D:/Users/Jesus CabreraAlvarez/PycharmProjects/neuroStimulation/PSE/", folder, sep=""))
  
  # Gather FC data
  files_fc=dir(pattern="FC_ACC&Pr.csv")
  table=data.frame()
  
  # Extract and create main and needed variables
  for (f in files_fc){
    
    file_fft=dir(pattern = paste(strtrim(f, 9),"-FFT_ACC&Pr.csv", sep = ""))
  
    table_fft = read.csv(file_fft)[,c(1,4,5,6,7)]
    table_fft_avg = aggregate(table_fft[, 2:5], list(table_fft$stimFreq), mean)
    
    table_fc = read.csv(f)[,c(1,4,5,6,7)] ## import the csv without index
    table_fc_avg = aggregate(table_fc[, 2:5], list(table_fc$stimFreq), mean)
    
    table_merged = cbind(table_fc_avg, table_fft_avg[,2:5])
    colnames(table_merged)[1] = "stimFreq"
    
    # By now just prL
    table_merged["alphaPeak"] =  mean(c(table_merged$Precuneus_L[1], table_merged$Precuneus_R[1]))
    table_merged["stimF_rel"] = table_merged$stimFreq - table_merged$alphaPeak
    table_merged = table_merged %>% mutate_at(vars(stimFreq, stimF_rel),funs(round(.,2)))
    
    # By now just rel1
    table_merged["deltaFC_accLprL"] = table_merged$ACC_L.Precuneus_L - table_merged$ACC_L.Precuneus_L[1]
    table_merged["deltaFC_accRprL"] = table_merged$ACC_R.Precuneus_L - table_merged$ACC_R.Precuneus_L[1]
    table_merged["deltaFC_accLprR"] = table_merged$ACC_L.Precuneus_R - table_merged$ACC_L.Precuneus_R[1]
    table_merged["deltaFC_accRprR"] = table_merged$ACC_R.Precuneus_R - table_merged$ACC_R.Precuneus_R[1]
    
    table_merged["subj"] = strtrim(f, 9)
    
    table = rbind(table, table_merged)
    
  }
  
  ## Work by now on one rel: ACCl - Prl
  
  # Nos interesa saber a que distancia relativa del pico alpha debemos estimular
  # para ello, necesitamos ajustar los picos de los distintos sujetos. 
  
  # para prL
  diff_peaks = max(table$alphaPeak) - min(table$alphaPeak)
  
  # DATA: Clean the table to get a subset with comparable observations and make bins
  data = table[table$stimFreq!=0.0, c("stimF_rel","deltaFC_accRprL","deltaFC_accLprL","deltaFC_accLprR","deltaFC_accRprR","subj")]
  data = data[min(data$stimF_rel) + diff_peaks + 1 < data$stimF_rel, ] # "+/-1 to Reduce the number of bins to analise
  data = data[data$stimF_rel < max(data$stimF_rel) - diff_peaks - 1, ]
  
  data["stimF_relbin"] = cut(data$stimF_rel, ceiling(length(data$stimF_rel)/10), dig.lab=2)
  
  # revisa ahora los bins para asegurarte de que en cada bin haya 10 sujetos diferentes, elimina el resto
  for (bin in levels(data$stimF_relbin)){
    
    subj_list=data$subj[data$stimF_relbin==bin]
    
    bin_max = 
    bin_min = 
    
    if (any(duplicated(subj_list)) | length(subj_list)!=10){
      print("Check this bin...")
      print(bin)
      print(subj_list)
      # then delete that bin
      
    }
  }
  
  # From wide format to long
  data_long = gather(data, deltaFC_pair, deltaFC_value, deltaFC_accRprL:deltaFC_accRprR, factor_key=TRUE)

  
  # ggboxplot(rel1, x = "stimF_rel", y = "fc_diff", add = "jitter")
  
  # paired ttests for each stimF_rel vs reference
  
  for (rel in levels(data_long$deltaFC_pair)){
    
    data_subset = data_long[data_long$deltaFC_pair == rel, ]
    
    for (bin in levels(data_subset$stimF_relbin)){
      
      diff_dist = data_subset[data_subset$stimF_relbin==bin, "deltaFC_value"]
      
      stat = t.test(diff_dist, mu=0, alternative="two.sided")
      
      corr_pval = p.adjust(stat$p.value, method="fdr", n=length(levels(data_subset$stimF_relbin)))
      
      effect_size = stat$estimate/sd(diff_dist)
      
      if (stat$p.value<=0.001){
          signif = "***"
      } else if(stat$p.value<=0.01){
          signif = "**"
      } else if(stat$p.value<=0.05){
          signif = "*"
      } else{
          signif = ""
      }
      
      if (corr_pval<=0.001){
        corr_signif = "***"
      } else if(corr_pval<=0.01){
        corr_signif = "**"
      } else if(corr_pval<=0.05){
        corr_signif = "*"
      } else{
        corr_signif = ""
      }
      
      newRow = data.frame(bin, stat$parameter, stat$statistic, stat$p.value, signif, corr_pval, corr_signif, stat$estimate, effect_size, rel, weight)
      
      Results = rbind(Results, newRow)
      
    }
  }
  
  for (rel in levels(data_long$deltaFC_pair)){
    
    results_temp = Results[(Results$rel==rel) & (Results$weight==weight), ]
    data_subset = data_long[data_long$deltaFC_pair == rel, ]
    
      
    print(ggplot(data_subset, aes(x = stimF_relbin, y = deltaFC_value )) +
      geom_boxplot() +
      geom_point(aes(colour = subj)) + 
      theme(axis.text.x = element_text(angle=45, hjust=0.98, size=12)) +
      labs(x="Stimulation Frenquency relative to alpha peak",y="Delta FC",
           title = paste("Stimulation influence on FC between",
                         substr(rel,13,16), " - " , substr(rel,9,12),
                         " | stimW = ", weight)) + 
      ylim(c(-0.7,1)) +
      annotate("text", x=factor(results_temp$bin), 
               y=results_temp$stat.estimate + 0.5, 
               label = results_temp$corr_signif, angle=90))
    
    name=paste("deltaFC_",substr(rel,13,16), "-" , substr(rel,9,12), "mccorr.png", sep="")
    ggsave(file=name, width = 19.1, height = 10.16, units = "cm")
    
  }
  
  ggboxplot(data_long, x="stimF_relbin", y = "deltaFC_value")+
    rotate_x_text(angle=45)+
    geom_hline(yintercept = mean(data_long$deltaFC_value), linetype=2)+
    stat_compare_means(method="t.test", label = "p.signif", ref.group = ".all.", hide.ns =TRUE) +
    facet_grid(deltaFC_pair ~.)
  
  rm (table_fc, table_fc_avg, table_fft, table_fft_avg, table_merged, f, file_fft, files_fc)
  rm(data_subset, results_temp)
  rm(newRow, stat, diff_peaks, bin, subj_list, diff_dist)
  rm (data)

}









